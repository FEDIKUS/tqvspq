#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import json
import math
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Iterable

import faiss
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    base: str
    query: str
    ground_truth: str | None = None


@dataclasses.dataclass(frozen=True)
class BenchmarkConfig:
    pq_bits: int
    tq_bits: int
    topk: int
    methods: tuple[str, ...]
    pq_train_size: int | None
    base_batch_size: int
    query_batch_size: int
    workers: int
    faiss_threads: int
    torch_threads: int
    torch_interop_threads: int
    seed: int
    output: str | None


@dataclasses.dataclass
class RunningNormStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    max_value: float = 0.0

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return
        self.count += int(arr.size)
        self.total += float(arr.sum())
        self.total_sq += float(np.square(arr).sum())
        self.max_value = max(self.max_value, float(arr.max()))

    def to_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {"mean": 0.0, "std": 0.0, "rms": 0.0, "max": 0.0}
        mean = self.total / self.count
        variance = max(0.0, (self.total_sq / self.count) - mean * mean)
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "rms": math.sqrt(self.total_sq / self.count),
            "max": self.max_value,
        }


@dataclasses.dataclass
class DriftStats:
    count: int = 0
    signed_total: float = 0.0
    abs_total: float = 0.0
    sq_total: float = 0.0
    max_abs: float = 0.0

    def update(self, delta: np.ndarray) -> None:
        arr = np.asarray(delta, dtype=np.float64)
        if arr.size == 0:
            return
        abs_arr = np.abs(arr)
        self.count += int(arr.size)
        self.signed_total += float(arr.sum())
        self.abs_total += float(abs_arr.sum())
        self.sq_total += float(np.square(arr).sum())
        self.max_abs = max(self.max_abs, float(abs_arr.max()))

    def to_dict(self) -> dict[str, float]:
        if self.count == 0:
            return {
                "mean_signed_change": 0.0,
                "mean_absolute_change": 0.0,
                "rmse_change": 0.0,
                "max_absolute_change": 0.0,
            }
        return {
            "mean_signed_change": self.signed_total / self.count,
            "mean_absolute_change": self.abs_total / self.count,
            "rmse_change": math.sqrt(self.sq_total / self.count),
            "max_absolute_change": self.max_abs,
        }


def log(message: str) -> None:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def iter_slices(total: int, batch_size: int) -> Iterable[slice]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, total, batch_size):
        yield slice(start, min(total, start + batch_size))


def resolve_path(path: str, data_root: str | None, manifest_dir: pathlib.Path) -> str:
    candidate = pathlib.Path(path)
    if candidate.is_absolute():
        return str(candidate)
    if data_root:
        return str((pathlib.Path(data_root) / candidate).resolve())
    return str((manifest_dir / candidate).resolve())


def load_manifest(path: str, data_root: str | None) -> list[DatasetSpec]:
    manifest_path = pathlib.Path(path).resolve()
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(manifest_path.read_text())
    elif manifest_path.suffix.lower() == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML manifest requested but tomllib is not available")
        payload = tomllib.loads(manifest_path.read_text())
    else:
        raise ValueError("Manifest must be .json or .toml")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Manifest must contain a non-empty top-level 'datasets' array")

    manifest_dir = manifest_path.parent
    resolved: list[DatasetSpec] = []
    for item in datasets:
        resolved.append(
            DatasetSpec(
                name=str(item["name"]),
                base=resolve_path(str(item["base"]), data_root, manifest_dir),
                query=resolve_path(str(item["query"]), data_root, manifest_dir),
                ground_truth=resolve_path(str(item["ground_truth"]), data_root, manifest_dir)
                if item.get("ground_truth")
                else None,
            )
        )
    return resolved


def read_bin_header(path: str) -> tuple[int, int]:
    header = np.fromfile(path, dtype=np.int32, count=2)
    if header.size != 2:
        raise ValueError(f"Could not read (rows, dim) header from {path}")
    rows = int(header[0])
    dim = int(header[1])
    if rows <= 0 or dim <= 0:
        raise ValueError(f"Invalid header in {path}: rows={rows}, dim={dim}")
    return rows, dim


def open_vector_bin(path: str) -> np.memmap:
    rows, dim = read_bin_header(path)
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(rows, dim))


def slice_rows(matrix: np.memmap, slc: slice) -> np.ndarray:
    return np.array(matrix[slc], dtype=np.float32, order="C", copy=True)


def take_rows(matrix: np.memmap, indices: np.ndarray) -> np.ndarray:
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    if flat.size == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float32)
    if flat.min() < 0 or flat.max() >= matrix.shape[0]:
        raise ValueError("Ground-truth neighbor id is out of bounds")
    unique, inverse = np.unique(flat, return_inverse=True)
    rows = np.array(matrix[unique], dtype=np.float32, order="C", copy=True)
    return rows[inverse]


def load_queries(path: str) -> np.ndarray:
    return np.array(open_vector_bin(path), dtype=np.float32, order="C", copy=True)


def load_ground_truth_ids(path: str, query_count: int, topk: int) -> np.ndarray:
    gt_queries, gt_k = read_bin_header(path)
    if gt_queries < query_count:
        raise ValueError(f"{path} has only {gt_queries} queries, expected at least {query_count}")
    if gt_k < topk:
        raise ValueError(f"{path} has only top-{gt_k} neighbors, requested top-{topk}")

    ids = np.memmap(path, dtype=np.uint32, mode="r", offset=8, shape=(gt_queries, gt_k))
    return np.array(ids[:query_count, :topk], dtype=np.int64, order="C", copy=True)


def sample_training_vectors(
    base: np.memmap,
    bits: int,
    train_size: int | None,
    seed: int,
) -> np.ndarray:
    min_required = 1 << bits
    # FAISS warns below 39 training points per centroid for PQ k-means.
    recommended = max(4096, 39 * min_required)
    size = min(base.shape[0], train_size or recommended)
    if size < min_required:
        raise ValueError(
            f"Need at least {min_required} training vectors for {bits}-bit PQ, only {base.shape[0]} available"
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(base.shape[0], size=size, replace=False)
    return take_rows(base, indices)


def compute_exact_topk(
    base: np.memmap,
    query: np.ndarray,
    topk: int,
    base_batch_size: int,
    query_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if base.shape[0] < topk:
        raise ValueError(f"base_count={base.shape[0]} is smaller than requested topk={topk}")

    index = faiss.IndexFlatL2(base.shape[1])
    for slc in iter_slices(base.shape[0], base_batch_size):
        index.add(slice_rows(base, slc))

    distances = np.empty((query.shape[0], topk), dtype=np.float32)
    indices = np.empty((query.shape[0], topk), dtype=np.int64)
    for slc in iter_slices(query.shape[0], query_batch_size):
        batch_distances, batch_indices = index.search(query[slc], topk)
        distances[slc] = np.sqrt(np.maximum(batch_distances, 0.0))
        indices[slc] = batch_indices.astype(np.int64, copy=False)
    return indices, distances


def compute_reference_distances(
    base: np.memmap,
    query: np.ndarray,
    neighbor_ids: np.ndarray,
    query_batch_size: int,
) -> np.ndarray:
    topk = neighbor_ids.shape[1]
    distances = np.empty((query.shape[0], topk), dtype=np.float32)
    for slc in iter_slices(query.shape[0], query_batch_size):
        query_batch = query[slc]
        neighbors = take_rows(base, neighbor_ids[slc]).reshape(query_batch.shape[0], topk, base.shape[1])
        distances[slc] = np.linalg.norm(query_batch[:, None, :] - neighbors, axis=2)
    return distances


def configure_threads(config: BenchmarkConfig) -> None:
    faiss.omp_set_num_threads(config.faiss_threads)
    os.environ["OMP_NUM_THREADS"] = str(config.faiss_threads)

    if "turboquant_mse" not in config.methods:
        return

    import torch

    torch.set_num_threads(config.torch_threads)
    try:
        torch.set_num_interop_threads(config.torch_interop_threads)
    except RuntimeError:
        pass


def benchmark_reconstruction(
    base: np.memmap,
    batch_size: int,
    reconstruct_fn,
) -> RunningNormStats:
    stats = RunningNormStats()
    for slc in iter_slices(base.shape[0], batch_size):
        batch = slice_rows(base, slc)
        decoded = reconstruct_fn(batch)
        stats.update(np.linalg.norm(batch - decoded, axis=1))
    return stats


def benchmark_topk_drift(
    query: np.ndarray,
    decoded_neighbors: np.ndarray,
    reference_distances: np.ndarray,
    batch_size: int,
) -> DriftStats:
    stats = DriftStats()
    for slc in iter_slices(query.shape[0], batch_size):
        query_batch = query[slc]
        approx_distances = np.linalg.norm(query_batch[:, None, :] - decoded_neighbors[slc], axis=2)
        stats.update(approx_distances - reference_distances[slc])
    return stats


def predecode_neighbors(
    base: np.memmap,
    neighbor_ids: np.ndarray,
    batch_size: int,
    reconstruct_fn,
) -> np.ndarray:
    query_count, topk = neighbor_ids.shape
    unique_ids, inverse = np.unique(neighbor_ids.reshape(-1), return_inverse=True)
    decoded_unique = np.empty((unique_ids.shape[0], base.shape[1]), dtype=np.float32)
    for slc in iter_slices(unique_ids.shape[0], batch_size):
        batch_ids = unique_ids[slc]
        batch = take_rows(base, batch_ids)
        decoded_unique[slc] = reconstruct_fn(batch)
    return decoded_unique[inverse].reshape(query_count, topk, base.shape[1])


def benchmark_pq(
    base: np.memmap,
    query: np.ndarray,
    neighbor_ids: np.ndarray,
    reference_distances: np.ndarray,
    config: BenchmarkConfig,
    dataset_name: str,
) -> dict[str, Any]:
    dim = int(base.shape[1])
    pq = faiss.ProductQuantizer(dim, dim, config.pq_bits)

    training_vectors = sample_training_vectors(base, config.pq_bits, config.pq_train_size, config.seed)
    train_start = time.perf_counter()
    pq.train(training_vectors)
    train_seconds = time.perf_counter() - train_start
    log(f"{dataset_name}: PQ training finished in {train_seconds:.2f}s")

    def reconstruct(batch: np.ndarray) -> np.ndarray:
        return pq.decode(pq.compute_codes(batch))

    recon_start = time.perf_counter()
    recon_stats = benchmark_reconstruction(base, config.base_batch_size, reconstruct)
    recon_seconds = time.perf_counter() - recon_start
    log(f"{dataset_name}: PQ reconstruction pass finished in {recon_seconds:.2f}s")

    neighbor_decode_start = time.perf_counter()
    decoded_neighbors = predecode_neighbors(base, neighbor_ids, config.base_batch_size, reconstruct)
    neighbor_decode_seconds = time.perf_counter() - neighbor_decode_start
    log(f"{dataset_name}: PQ neighbor predecode finished in {neighbor_decode_seconds:.2f}s")

    drift_start = time.perf_counter()
    drift_stats = benchmark_topk_drift(
        query,
        decoded_neighbors,
        reference_distances,
        config.query_batch_size,
    )
    drift_seconds = time.perf_counter() - drift_start
    log(f"{dataset_name}: PQ top-k drift pass finished in {drift_seconds:.2f}s")

    total_seconds = train_seconds + recon_seconds + neighbor_decode_seconds + drift_seconds

    return {
        "method": "pq",
        "bits_per_coordinate": config.pq_bits,
        "subspace_size": 1,
        "bytes_per_vector": int(pq.code_size),
        "train_seconds": train_seconds,
        "reconstruction_seconds": recon_seconds,
        "neighbor_predecode_seconds": neighbor_decode_seconds,
        "topk_distance_seconds": drift_seconds,
        "total_seconds": total_seconds,
        "reconstruction_l2": recon_stats.to_dict(),
        "topk_distance_change_l2": drift_stats.to_dict(),
    }


def benchmark_turboquant_mse(
    base: np.memmap,
    query: np.ndarray,
    neighbor_ids: np.ndarray,
    reference_distances: np.ndarray,
    config: BenchmarkConfig,
    dataset_name: str,
) -> dict[str, Any]:
    import torch
    from turboquant import TurboQuantMSE

    quantizer = TurboQuantMSE(
        dim=base.shape[1],
        bits=config.tq_bits,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ).eval()

    with torch.no_grad():
        packed_len = int(quantizer.quantize(torch.from_numpy(slice_rows(base, slice(0, 1)))).indices.shape[1])

    def reconstruct(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.from_numpy(batch)
            quantized = quantizer.quantize(tensor)
            return quantizer.dequantize(quantized).cpu().numpy()

    recon_start = time.perf_counter()
    recon_stats = benchmark_reconstruction(base, config.base_batch_size, reconstruct)
    recon_seconds = time.perf_counter() - recon_start
    log(f"{dataset_name}: TurboQuant reconstruction pass finished in {recon_seconds:.2f}s")

    neighbor_decode_start = time.perf_counter()
    decoded_neighbors = predecode_neighbors(base, neighbor_ids, config.base_batch_size, reconstruct)
    neighbor_decode_seconds = time.perf_counter() - neighbor_decode_start
    log(f"{dataset_name}: TurboQuant neighbor predecode finished in {neighbor_decode_seconds:.2f}s")

    drift_start = time.perf_counter()
    drift_stats = benchmark_topk_drift(
        query,
        decoded_neighbors,
        reference_distances,
        config.query_batch_size,
    )
    drift_seconds = time.perf_counter() - drift_start
    log(f"{dataset_name}: TurboQuant top-k drift pass finished in {drift_seconds:.2f}s")

    total_seconds = recon_seconds + neighbor_decode_seconds + drift_seconds

    return {
        "method": "turboquant_mse",
        "bits_per_coordinate": config.tq_bits,
        "subspace_size": 1,
        "bytes_per_vector": int(packed_len + 4),
        "packed_index_bytes_per_vector": int(packed_len),
        "norm_bytes_per_vector": 4,
        "train_seconds": 0.0,
        "reconstruction_seconds": recon_seconds,
        "neighbor_predecode_seconds": neighbor_decode_seconds,
        "topk_distance_seconds": drift_seconds,
        "total_seconds": total_seconds,
        "reconstruction_l2": recon_stats.to_dict(),
        "topk_distance_change_l2": drift_stats.to_dict(),
    }


def benchmark_dataset(dataset: DatasetSpec, config: BenchmarkConfig) -> dict[str, Any]:
    configure_threads(config)

    log(f"{dataset.name}: opening base vectors")
    base = open_vector_bin(dataset.base)
    log(f"{dataset.name}: loading queries")
    query = load_queries(dataset.query)

    if query.shape[1] != base.shape[1]:
        raise ValueError(
            f"Dimension mismatch for {dataset.name}: base={base.shape[1]} query={query.shape[1]}"
        )

    gt_start = time.perf_counter()
    if dataset.ground_truth:
        log(f"{dataset.name}: loading top-{config.topk} ground-truth neighbor ids")
        neighbor_ids = load_ground_truth_ids(dataset.ground_truth, query.shape[0], config.topk)
        reference_distances = compute_reference_distances(base, query, neighbor_ids, config.query_batch_size)
        topk_source = "ground_truth"
    else:
        log(f"{dataset.name}: computing exact top-{config.topk} neighbors with Faiss")
        neighbor_ids, reference_distances = compute_exact_topk(
            base,
            query,
            config.topk,
            config.base_batch_size,
            config.query_batch_size,
        )
        topk_source = "exact_search"
    gt_seconds = time.perf_counter() - gt_start

    methods: dict[str, Any] = {}
    if "pq" in config.methods:
        log(f"{dataset.name}: running PQ ({config.pq_bits} bits/coord)")
        methods["pq"] = benchmark_pq(base, query, neighbor_ids, reference_distances, config, dataset.name)
    if "turboquant_mse" in config.methods:
        log(f"{dataset.name}: running TurboQuant MSE ({config.tq_bits} bits/coord)")
        methods["turboquant_mse"] = benchmark_turboquant_mse(
            base,
            query,
            neighbor_ids,
            reference_distances,
            config,
            dataset.name,
        )

    return {
        "dataset": dataset.name,
        "base_count": int(base.shape[0]),
        "query_count": int(query.shape[0]),
        "dimension": int(base.shape[1]),
        "topk": config.topk,
        "topk_source": topk_source,
        "topk_reference_seconds": gt_seconds,
        "methods": methods,
    }


def benchmark_dataset_entry(dataset_payload: dict[str, Any], config_payload: dict[str, Any]) -> dict[str, Any]:
    dataset = DatasetSpec(**dataset_payload)
    config = BenchmarkConfig(**config_payload)
    try:
        return benchmark_dataset(dataset, config)
    except Exception as exc:  # pragma: no cover
        return {
            "dataset": dataset.name,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def summarize(results: list[dict[str, Any]]) -> None:
    print()
    for result in results:
        if "error" in result:
            print(f"{result['dataset']}: ERROR: {result['error']}")
            continue
        print(
            f"{result['dataset']}  nbase={result['base_count']}  "
            f"nquery={result['query_count']}  dim={result['dimension']}  "
            f"topk_source={result['topk_source']}"
        )
        for method_name, payload in result["methods"].items():
            recon = payload["reconstruction_l2"]
            drift = payload["topk_distance_change_l2"]
            print(
                "  "
                f"{method_name}: bits={payload['bits_per_coordinate']} "
                f"train_s={payload['train_seconds']:.2f} "
                f"recon_s={payload['reconstruction_seconds']:.2f} "
                f"neighbor_s={payload['neighbor_predecode_seconds']:.2f} "
                f"drift_s={payload['topk_distance_seconds']:.2f} "
                f"total_s={payload['total_seconds']:.2f} "
                f"recon_mean={recon['mean']:.6f} "
                f"topk_abs_mean={drift['mean_absolute_change']:.6f} "
                f"topk_signed_mean={drift['mean_signed_change']:.6f}"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU benchmark for FAISS PQ and TurboQuant MSE on headered float32 .bin datasets."
    )
    parser.add_argument("--manifest", required=True, help="JSON or TOML file listing datasets")
    parser.add_argument("--data-root", default=None, help="Optional prefix for relative paths in the manifest")
    parser.add_argument("--pq-bits", type=int, default=8, help="Bits per coordinate for FAISS PQ")
    parser.add_argument("--tq-bits", type=int, default=3, help="Bits per coordinate for TurboQuant MSE")
    parser.add_argument("--topk", type=int, default=20, help="Number of neighbors used for the distance-change metric")
    parser.add_argument(
        "--methods",
        default="pq,turboquant_mse",
        help="Comma-separated methods to run: pq,turboquant_mse",
    )
    parser.add_argument(
        "--pq-train-size",
        type=int,
        default=None,
        help="Optional PQ training sample size; default is based on bit-width",
    )
    parser.add_argument("--base-batch-size", type=int, default=16384, help="Base-vector batch size")
    parser.add_argument("--query-batch-size", type=int, default=512, help="Query batch size")
    parser.add_argument("--workers", type=int, default=1, help="Parallel dataset workers")
    parser.add_argument("--faiss-threads", type=int, default=None, help="FAISS CPU threads per worker")
    parser.add_argument("--torch-threads", type=int, default=None, help="Torch CPU threads per worker")
    parser.add_argument("--torch-interop-threads", type=int, default=1, help="Torch inter-op threads per worker")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for PQ training sampling")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def resolve_thread_defaults(args: argparse.Namespace, dataset_count: int) -> tuple[int, int, int, int]:
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(args.workers, dataset_count))
    per_worker = max(1, cpu_count // workers)
    faiss_threads = args.faiss_threads or per_worker
    torch_threads = args.torch_threads or per_worker
    return workers, faiss_threads, max(1, torch_threads), max(1, args.torch_interop_threads)


def main() -> int:
    args = parse_args()
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    allowed = {"pq", "turboquant_mse"}
    unknown = sorted(set(methods) - allowed)
    if not methods:
        raise ValueError("At least one method must be specified")
    if unknown:
        raise ValueError(f"Unsupported methods: {', '.join(unknown)}")

    datasets = load_manifest(args.manifest, args.data_root)
    workers, faiss_threads, torch_threads, torch_interop_threads = resolve_thread_defaults(args, len(datasets))

    config = BenchmarkConfig(
        pq_bits=args.pq_bits,
        tq_bits=args.tq_bits,
        topk=args.topk,
        methods=methods,
        pq_train_size=args.pq_train_size,
        base_batch_size=args.base_batch_size,
        query_batch_size=args.query_batch_size,
        workers=workers,
        faiss_threads=faiss_threads,
        torch_threads=torch_threads,
        torch_interop_threads=torch_interop_threads,
        seed=args.seed,
        output=args.output,
    )

    log(
        "Configuration: "
        f"workers={config.workers}, faiss_threads={config.faiss_threads}, "
        f"torch_threads={config.torch_threads}, methods={','.join(config.methods)}"
    )

    started_at = dt.datetime.now(dt.timezone.utc)
    dataset_payloads = [dataclasses.asdict(dataset) for dataset in datasets]
    config_payload = dataclasses.asdict(config)

    if config.workers == 1:
        results = [benchmark_dataset_entry(dataset_payload, config_payload) for dataset_payload in dataset_payloads]
    else:
        results = []
        mp_context = __import__("multiprocessing").get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=config.workers,
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(benchmark_dataset_entry, dataset_payload, config_payload)
                for dataset_payload in dataset_payloads
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["dataset"])

    finished_at = dt.datetime.now(dt.timezone.utc)
    summary = {
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "config": config_payload,
        "results": results,
    }

    summarize(results)

    if config.output:
        output_path = pathlib.Path(config.output)
        output_path.write_text(json.dumps(summary, indent=2))
        log(f"Wrote JSON results to {output_path}")
    else:
        print(json.dumps(summary, indent=2))

    return 1 if any("error" in item for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
