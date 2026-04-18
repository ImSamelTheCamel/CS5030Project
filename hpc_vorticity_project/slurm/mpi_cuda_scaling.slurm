#!/bin/bash
#SBATCH --job-name=vort_mpi_cuda_scale
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=4
#SBATCH --ntasks=4
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
BLOCK_X=16
BLOCK_Y=16
REPEATS=5
RANKS_LIST=(2 4)

[[ -x ./mpi_cuda_vorticity ]] || nvcc -ccbin mpic++ -O2 -std=c++11 mpi_cuda_vorticity.cu -o mpi_cuda_vorticity

OUTFILE="results/mpi_cuda_scaling_${SLURM_JOB_ID}.csv"
echo "experiment,impl,nodes,ranks,run,block_x,block_y,local_kernel_runtime_ms,notes" > "$OUTFILE"

for ranks in "${RANKS_LIST[@]}"; do
  nodes="$ranks"
  for run in $(seq 1 "$REPEATS"); do
    mpi_cuda_out=$(mpirun -np "$ranks" --map-by ppr:1:node ./mpi_cuda_vorticity --input "$INPUT" --output-prefix "mpi_cuda_n${nodes}_r${ranks}_run${run}" --width "$WIDTH" --height "$HEIGHT" --block-x "$BLOCK_X" --block-y "$BLOCK_Y")
    mpi_cuda_rt=$(echo "$mpi_cuda_out" | awk '/MPI CUDA local kernel runtime \(ms\):/ {print $7}' | head -n 1)
    echo "mpi_cuda_scaling,mpi_cuda,${nodes},${ranks},${run},${BLOCK_X},${BLOCK_Y},${mpi_cuda_rt},local_kernel_time" >> "$OUTFILE"
  done
done

echo "Wrote $OUTFILE"
