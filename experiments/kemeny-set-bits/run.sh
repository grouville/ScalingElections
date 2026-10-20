#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
repository_dir=$(git -C "$script_dir" rev-parse --show-toplevel)
cd -- "$script_dir"
printf 'GPU experiment source commit:\n'
git -C "$repository_dir" rev-parse HEAD
nvcc --version
nvidia-smi -L
python3 make_gpu_variant.py "$repository_dir/kemeny.cuh"
sha256sum "$repository_dir/kemeny.cuh" gpu-set-bits/kemeny.cuh
options=(-std=c++20 -O3 --expt-relaxed-constexpr -arch=native -Xcompiler -fopenmp -I "$repository_dir")
printf 'Compiling baseline GPU entry point...\n'
nvcc "${options[@]}" -DENTRY_POINT=gpu_baseline -DCOSTS_ENTRY=gpu_costs_baseline \
    -Dkemeny_layer_cuda_=kemeny_layer_baseline -Dcompute_kemeny_costs_gpu=kemeny_costs_baseline \
    -Dcompute_kemeny_ranking=kemeny_ranking_baseline -c kemeny_gpu_entry.cu -o baseline.o
printf 'Compiling GPU set-bit traversal...\n'
nvcc -I "$script_dir/gpu-set-bits" "${options[@]}" -DENTRY_POINT=gpu_changed -DCOSTS_ENTRY=gpu_costs_changed \
    -Dkemeny_layer_cuda_=kemeny_layer_changed -Dcompute_kemeny_costs_gpu=kemeny_costs_changed \
    -Dcompute_kemeny_ranking=kemeny_ranking_changed -c kemeny_gpu_entry.cu -o changed.o
nvcc "${options[@]}" kemeny_gpu_pairs.cu baseline.o changed.o -o gpu-pairs
printf 'Running GPU correctness checks and paired complete solves...\n'
OMP_NUM_THREADS=4 timeout 600 ./gpu-pairs > results.csv
python3 summarize_gpu_kemeny.py results.csv
if command -v compute-sanitizer >/dev/null 2>&1; then
    printf 'Checking both GPU kernels with Compute Sanitizer...\n'
    OMP_NUM_THREADS=4 timeout 180 compute-sanitizer --tool memcheck --error-exitcode 3 ./gpu-pairs --verify-only
else
    printf 'SKIP: Compute Sanitizer is unavailable.\n'
fi
printf 'Raw samples and summary: %s/results.csv and %s/results.summary.json\n' "$script_dir" "$script_dir"
