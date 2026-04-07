# ANN Quantization Benchmark

This benchmark does two things on CPU:

- measures reconstruction L2 between each original base vector and its quantized-then-dequantized version
- measures how much the L2 distance changes, on average, between each query and its top-`k` neighbors

It compares:

- FAISS PQ with subspace size `1`
- TurboQuant MSE from `https://github.com/0xSero/turboquant.git`

The script is intentionally narrow:

- vectors are read from `.bin` files with an 8-byte `(rows, dim)` header followed by `float32` payload
- top-`k` neighbors come from a matching `.bin` ground-truth file when provided
- if a ground-truth file also contains stored distances, the script ignores them and recomputes L2 distances from the original vectors
- if no ground-truth file is provided, the script falls back to exact CPU search with Faiss

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Manifest

Use a JSON or TOML file with entries like this:

```json
{
  "datasets": [
    {
      "name": "gist1m",
      "base": "gist1M960/base.bin",
      "query": "gist1M960/query.bin",
      "ground_truth": "gist1M960/gt.bin"
    }
  ]
}
```

If the paths are relative, pass `--data-root /path/to/datasets`.

## Usage

```bash
python ann_benchmark.py \
  --manifest datasets_manifest.json \
  --data-root /path/to/datasets \
  --pq-bits 8 \
  --tq-bits 3 \
  --topk 20 \
  --workers 2 \
  --output results.json
```

Important notes:

- `--pq-bits` and `--tq-bits` are independent
- `--topk` defaults to `20`
- large datasets should use `ground_truth` so the script does not have to brute-force exact top-`k`
- `1M4096/gt.bin` only contains top-10, so that dataset cannot be run with `--topk 20` unless you compute exact neighbors separately
