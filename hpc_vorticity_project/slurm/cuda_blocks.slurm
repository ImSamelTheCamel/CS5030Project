#!/bin/bash
#SBATCH --job-name=vort_cuda_blocks
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

set -euo pipefail

mkdir -p logs results

module purge || true
module load gcc
module load openmpi
module load cuda

INPUT="../cyl2d_1300x600_float32[2].raw"
WIDTH=1300
HEIGHT=600
REPEATS=5
BLOCKS=("8 8" "16 16" "32 8" "32 16")

[[ -x ./cuda_vorticity ]] || nvcc -O2 -std=c++11 cuda_vorticity.cu -o cuda_vorticity

OUTFILE="results/cuda_blocks_${SLURM_JOB_ID}.csv"
echo "experiment,impl,block_x,block_y,run,runtime_ms,notes" > "$OUTFILE"

for b in "${BLOCKS[@]}"; do
  bx=$(echo "$b" | awk '{print $1}')
  by=$(echo "$b" | awk '{print $2}')
  for run in $(seq 1 "$REPEATS"); do
    cuda_out=$(./cuda_vorticity --input "$INPUT" --output-prefix "cuda_b${bx}x${by}_r${run}" --width "$WIDTH" --height "$HEIGHT" --block-x "$bx" --block-y "$by")
    cuda_rt=$(echo "$cuda_out" | awk '/CUDA kernel runtime \(ms\):/ {print $5}')
    echo "cuda_blocks,cuda,${bx},${by},${run},${cuda_rt},kernel_only" >> "$OUTFILE"
  done

done

echo "Wrote $OUTFILE"
