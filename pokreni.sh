# tq

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --pq-bits 8 --output output-top-pq-100-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --pq-bits 7 --output output-top-pq-100-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --pq-bits 6 --output output-top-pq-100-6.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --pq-bits 8 --output output-top-pq-50-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --pq-bits 7 --output output-top-pq-50-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --pq-bits 6 --output output-top-pq-50-6.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --pq-bits 8 --output output-top-pq-20-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --pq-bits 7 --output output-top-pq-20-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method pq --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --pq-bits 6 --output output-top-pq-20-6.json

# pq

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --tq-bits 8 --output output-top-tq-100-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --tq-bits 7 --output output-top-tq-100-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 100 --tq-bits 6 --output output-top-tq-100-6.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --tq-bits 8 --output output-top-tq-50-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --tq-bits 7 --output output-top-tq-50-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 50 --tq-bits 6 --output output-top-tq-50-6.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --tq-bits 8 --output output-top-tq-20-8.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --tq-bits 7 --output output-top-tq-20-7.json

python ann_benchmark2.py --manifest datasets_manifest.json --data-root ../datasets --method turboquant_package --pq-train-size 1000000 --compare-topk 10 --faiss-threads 16 --torch-threads 16 --candidate-topx 20 --tq-bits 6 --output output-top-tq-20-6.json


