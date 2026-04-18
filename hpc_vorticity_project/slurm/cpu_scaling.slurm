#!/bin/bash
#SBATCH --job-name=vort_cpu_scale
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=kingspeak
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out

set -euo pipefail

mkdir -p logs results

module purge || true
module load gcc
module load openmpi

INPUT="../cyl2d_1300x600_float32[2].raw"
WIDTH=1300
HEIGHT=600
REPEATS=5
THREADS_LIST=(1 2 4 8 16)

[[ -x ./serial_vorticity ]] || g++ -std=c++11 -O2 serial_vorticity.cpp -o serial_vorticity
[[ -x ./openmp_vorticity ]] || g++ -std=c++11 -O2 -fopenmp openmp_vorticity.cpp -o openmp_vorticity

OUTFILE="results/cpu_scaling_${SLURM_JOB_ID}.csv"
echo "experiment,impl,threads,run,runtime_s,notes" > "$OUTFILE"

for run in $(seq 1 "$REPEATS"); do
  serial_out=$(./serial_vorticity --input "$INPUT" --output-prefix "serial_t1_r${run}" --width "$WIDTH" --height "$HEIGHT")
  serial_rt=$(echo "$serial_out" | awk '/Serial runtime \(s\):/ {print $4}')
  echo "cpu_scaling,serial,1,${run},${serial_rt},baseline" >> "$OUTFILE"
done

for t in "${THREADS_LIST[@]}"; do
  export OMP_NUM_THREADS="$t"
  for run in $(seq 1 "$REPEATS"); do
    omp_out=$(./openmp_vorticity --input "$INPUT" --output-prefix "omp_t${t}_r${run}" --width "$WIDTH" --height "$HEIGHT")
    omp_rt=$(echo "$omp_out" | awk '/OpenMP runtime \(s\):/ {print $4}')
    echo "cpu_scaling,openmp,${t},${run},${omp_rt},OMP_NUM_THREADS=${t}" >> "$OUTFILE"
  done
done

echo "Wrote $OUTFILE"
