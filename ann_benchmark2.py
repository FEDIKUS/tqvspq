#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

import faiss
import numpy as np

import ann_benchmark as shared


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def configure_threads(method: str, faiss_threads: int, torch_threads: int, torch_interop_threads: int) -> None:
    faiss.omp_set_num_threads(faiss_threads)
    os.environ["OMP_NUM_THREADS"] = str(faiss_threads)
    if method != "turboquant_package":
        return

    import torch

    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        pass


def make_pq_reconstruct_fn(base: np.memmap, bits: int, train_size: int | None, seed: int):
    dim = int(base.shape[1])
    pq = faiss.ProductQuantizer(dim, dim, bits)
    training_vectors = shared.sample_training_vectors(base, bits, train_size, seed)
    train_started = time.perf_counter()
    pq.train(training_vectors)
    train_seconds = time.perf_counter() - train_started

    def reconstruct(batch: np.ndarray) -> np.ndarray:
        return pq.decode(pq.compute_codes(batch))

    return reconstruct, {
        "method": "pq",
        "bits_per_coordinate": bits,
        "bytes_per_vector": int(pq.code_size),
        "subspace_size": 1,
        "train_seconds": train_seconds,
    }


def make_turboquant_package_reconstruct_fn(bits: int, dim: int, seed: int):
    import torch
    from turboquant import TurboQuantMSE

    quantizer = TurboQuantMSE(
        dim=dim,
        bits=bits,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=seed,
    ).eval()

    with torch.no_grad():
        sample = torch.zeros((1, dim), dtype=torch.float32)
        packed_len = int(quantizer.quantize(sample).indices.shape[1])

    def reconstruct(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(batch)
            quantized = quantizer.quantize(tensor)
            return quantizer.dequantize(quantized).cpu().numpy()

    return reconstruct, {
        "method": "turboquant_package",
        "bits_per_coordinate": bits,
        "bytes_per_vector": int(packed_len + 4),
        "packed_index_bytes_per_vector": int(packed_len),
        "norm_bytes_per_vector": 4,
        "subspace_size": 1,
        "train_seconds": 0.0,
    }


def decode_unique_candidates(
    base: np.memmap,
    candidate_ids: np.ndarray,
    batch_size: int,
    reconstruct_fn,
) -> tuple[np.ndarray, np.ndarray]:
    unique_ids, inverse = np.unique(candidate_ids.reshape(-1), return_inverse=True)
    decoded_unique = np.empty((unique_ids.shape[0], base.shape[1]), dtype=np.float32)
    for slc in shared.iter_slices(unique_ids.shape[0], batch_size):
        decoded_unique[slc] = reconstruct_fn(shared.take_rows(base, unique_ids[slc]))
    return decoded_unique, inverse.reshape(candidate_ids.shape)


def rerank_and_compare(
    query: np.ndarray,
    candidate_ids: np.ndarray,
    gt_top_ids: np.ndarray,
    decoded_unique: np.ndarray,
    candidate_inverse: np.ndarray,
    candidate_topx: int,
    compare_topk: int,
    query_batch_size: int,
) -> dict[str, float]:
    total_hits = 0
    full_recovery = 0
    exact_order = 0

    rerank_started = time.perf_counter()
    for slc in shared.iter_slices(query.shape[0], query_batch_size):
        query_batch = query[slc]
        batch_candidates = candidate_ids[slc]
        batch_inverse = candidate_inverse[slc].reshape(-1)
        decoded_batch = decoded_unique[batch_inverse].reshape(query_batch.shape[0], candidate_topx, query.shape[1])

        approx_distances = np.linalg.norm(query_batch[:, None, :] - decoded_batch, axis=2)
        order = np.argsort(approx_distances, axis=1, kind="stable")[:, :compare_topk]
        predicted_top = np.take_along_axis(batch_candidates, order, axis=1)
        gt_top = gt_top_ids[slc]

        for row_idx in range(query_batch.shape[0]):
            gt_row = gt_top[row_idx]
            pred_row = predicted_top[row_idx]
            overlap = len(set(gt_row.tolist()) & set(pred_row.tolist()))
            total_hits += overlap
            if overlap == compare_topk:
                full_recovery += 1
            if np.array_equal(gt_row, pred_row):
                exact_order += 1

    rerank_seconds = time.perf_counter() - rerank_started
    query_count = int(query.shape[0])
    denom = max(1, query_count * compare_topk)
    return {
        "rerank_seconds": rerank_seconds,
        "recall_percent": 100.0 * total_hits / denom,
        "full_recovery_percent": 100.0 * full_recovery / max(1, query_count),
        "exact_order_match_percent": 100.0 * exact_order / max(1, query_count),
    }


def benchmark_dataset(
    dataset: shared.DatasetSpec,
    *,
    method: str,
    pq_bits: int,
    tq_bits: int,
    candidate_topx: int,
    compare_topk: int,
    pq_train_size: int | None,
    base_batch_size: int,
    query_batch_size: int,
    faiss_threads: int,
    torch_threads: int,
    torch_interop_threads: int,
    seed: int,
) -> dict[str, Any]:
    if dataset.ground_truth is None:
        raise ValueError(f"{dataset.name}: ann_benchmark2 requires a ground-truth file")

    configure_threads(method, faiss_threads, torch_threads, torch_interop_threads)

    base = shared.open_vector_bin(dataset.base)
    query = shared.load_queries(dataset.query)
    if query.shape[1] != base.shape[1]:
        raise ValueError(f"{dataset.name}: dimension mismatch base={base.shape[1]} query={query.shape[1]}")

    requested_gt = max(candidate_topx, compare_topk)
    gt_started = time.perf_counter()
    all_gt_ids, effective_gt_topk = shared.load_ground_truth_ids(dataset.ground_truth, query.shape[0], requested_gt)
    gt_seconds = time.perf_counter() - gt_started

    effective_candidate_topx = min(candidate_topx, effective_gt_topk)
    effective_compare_topk = min(compare_topk, effective_candidate_topx)
    if effective_compare_topk <= 0:
        raise ValueError(f"{dataset.name}: no ground-truth neighbors available")

    candidate_ids = all_gt_ids[:, :effective_candidate_topx]
    gt_top_ids = all_gt_ids[:, :effective_compare_topk]

    if method == "pq":
        reconstruct_fn, method_info = make_pq_reconstruct_fn(base, pq_bits, pq_train_size, seed)
    elif method == "turboquant_package":
        reconstruct_fn, method_info = make_turboquant_package_reconstruct_fn(tq_bits, int(base.shape[1]), seed)
    else:
        raise ValueError(f"Unsupported method: {method}")

    decode_started = time.perf_counter()
    decoded_unique, candidate_inverse = decode_unique_candidates(base, candidate_ids, base_batch_size, reconstruct_fn)
    decode_seconds = time.perf_counter() - decode_started

    metrics = rerank_and_compare(
        query,
        candidate_ids,
        gt_top_ids,
        decoded_unique,
        candidate_inverse,
        effective_candidate_topx,
        effective_compare_topk,
        query_batch_size,
    )

    total_seconds = gt_seconds + method_info["train_seconds"] + decode_seconds + metrics["rerank_seconds"]
    return {
        "dataset": dataset.name,
        "base_count": int(base.shape[0]),
        "query_count": int(query.shape[0]),
        "dimension": int(base.shape[1]),
        "candidate_topx_requested": candidate_topx,
        "candidate_topx_used": effective_candidate_topx,
        "compare_topk_requested": compare_topk,
        "compare_topk_used": effective_compare_topk,
        "ground_truth_load_seconds": gt_seconds,
        "decode_candidates_seconds": decode_seconds,
        "total_seconds": total_seconds,
        **method_info,
        **metrics,
    }


def summarize(results: list[dict[str, Any]]) -> None:
    print()
    for result in results:
        if "error" in result:
            print(f"{result['dataset']}: ERROR: {result['error']}")
            continue
        print(
            f"{result['dataset']}  "
            f"topx={result['candidate_topx_used']}/{result['candidate_topx_requested']}  "
            f"topk={result['compare_topk_used']}/{result['compare_topk_requested']}  "
            f"method={result['method']}  "
            f"bits={result['bits_per_coordinate']}  "
            f"recall={result['recall_percent']:.4f}  "
            f"full={result['full_recovery_percent']:.4f}  "
            f"exact_order={result['exact_order_match_percent']:.4f}  "
            f"train_s={result['train_seconds']:.2f}  "
            f"decode_s={result['decode_candidates_seconds']:.2f}  "
            f"rerank_s={result['rerank_seconds']:.2f}  "
            f"total_s={result['total_seconds']:.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank ground-truth top-x candidates with PQ or TurboQuant and compare against the true top-10."
    )
    parser.add_argument("--manifest", required=True, help="JSON or TOML dataset manifest")
    parser.add_argument("--data-root", default=None, help="Optional prefix for relative paths in the manifest")
    parser.add_argument(
        "--method",
        choices=("pq", "turboquant_package"),
        required=True,
        help="Quantization method used for reranking",
    )
    parser.add_argument("--candidate-topx", type=int, required=True, help="Take the top-x ground-truth candidates")
    parser.add_argument("--compare-topk", type=int, default=10, help="Compare reranked top-k against true top-k")
    parser.add_argument("--pq-bits", type=int, default=8, help="Bits per coordinate for PQ")
    parser.add_argument("--tq-bits", type=int, default=8, help="Bits per coordinate for TurboQuant package")
    parser.add_argument("--pq-train-size", type=int, default=None, help="Optional PQ training sample size")
    parser.add_argument("--base-batch-size", type=int, default=2048, help="Base decode batch size")
    parser.add_argument("--query-batch-size", type=int, default=256, help="Query scoring batch size")
    parser.add_argument("--faiss-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--torch-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_topx <= 0:
        raise ValueError("--candidate-topx must be positive")
    if args.compare_topk <= 0:
        raise ValueError("--compare-topk must be positive")

    datasets = shared.load_manifest(args.manifest, args.data_root)
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        log(f"{dataset.name}: running {args.method}")
        try:
            results.append(
                benchmark_dataset(
                    dataset,
                    method=args.method,
                    pq_bits=args.pq_bits,
                    tq_bits=args.tq_bits,
                    candidate_topx=args.candidate_topx,
                    compare_topk=args.compare_topk,
                    pq_train_size=args.pq_train_size,
                    base_batch_size=args.base_batch_size,
                    query_batch_size=args.query_batch_size,
                    faiss_threads=args.faiss_threads,
                    torch_threads=args.torch_threads,
                    torch_interop_threads=args.torch_interop_threads,
                    seed=args.seed,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "dataset": dataset.name,
                    "error": str(exc),
                }
            )

    payload = {
        "config": {
            "method": args.method,
            "candidate_topx": args.candidate_topx,
            "compare_topk": args.compare_topk,
            "pq_bits": args.pq_bits,
            "tq_bits": args.tq_bits,
            "pq_train_size": args.pq_train_size,
            "base_batch_size": args.base_batch_size,
            "query_batch_size": args.query_batch_size,
            "faiss_threads": args.faiss_threads,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
            "seed": args.seed,
        },
        "results": results,
    }

    summarize(results)
    if args.output:
        output_path = pathlib.Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
