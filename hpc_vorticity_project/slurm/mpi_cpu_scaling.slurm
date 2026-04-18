#!/bin/bash
#SBATCH --job-name=vort_mpi_cpu_scale
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=kingspeak
#SBATCH --nodes=4
#SBATCH --ntasks=4
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
RANKS_LIST=(2 4)

[[ -x ./mpi_vorticity ]] || mpic++ -std=c++11 -O2 mpi_vorticity.cpp -o mpi_vorticity

OUTFILE="results/mpi_cpu_scaling_${SLURM_JOB_ID}.csv"
echo "experiment,impl,nodes,ranks,run,runtime_s,notes" > "$OUTFILE"

for ranks in "${RANKS_LIST[@]}"; do
  nodes="$ranks"
  for run in $(seq 1 "$REPEATS"); do
    mpi_out=$(mpirun -np "$ranks" --map-by ppr:1:node ./mpi_vorticity --input "$INPUT" --output-prefix "mpi_cpu_n${nodes}_r${ranks}_run${run}" --width "$WIDTH" --height "$HEIGHT")
    mpi_rt=$(echo "$mpi_out" | awk '/MPI CPU runtime \(s\):/ {print $5}')
    echo "mpi_cpu_scaling,mpi_cpu,${nodes},${ranks},${run},${mpi_rt},mpirun nodes=${nodes}" >> "$OUTFILE"
  done
done

echo "Wrote $OUTFILE"
