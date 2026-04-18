# HPC Vorticity Project

This project implements **Things are turning (vector field / vorticity analysis)** in separate files for:
- serial CPU
- parallel shared-memory CPU with OpenMP
- parallel CUDA GPU
- distributed-memory CPU with MPI
- distributed-memory GPU with MPI + CUDA

The supplied starter vorticity function is reused without modification in the CPU versions, and the same logic is mirrored in the GPU kernels.

## Files

- `vorticity_starter.hpp` - starter function copied exactly as given
- `common.hpp` - shared argument parsing, raw I/O, timing helper, magnitude helper
- `serial_vorticity.cpp` - serial CPU version
- `openmp_vorticity.cpp` - shared-memory CPU version
- `cuda_vorticity.cu` - single-GPU CUDA version
- `mpi_vorticity.cpp` - distributed-memory CPU version using MPI row decomposition and halo exchange
- `mpi_cuda_vorticity.cu` - distributed-memory GPU version using MPI row decomposition and one GPU per rank
- `validate.py` - validates a parallel vorticity output against the serial result
- `visualize.py` - saves PNG images for vorticity and magnitude outputs

## Input format

The input raw file is expected to contain `width * height * 2` float32 values.
Each grid cell stores two float32 values `(u, v)`.

Default dimensions in the code:
- width = 1300
- height = 600

## Build instructions on CHPC

### Suggested environments

For a CPU-only build on CHPC:
```bash
module purge
module load gcc
module load openmpi
```

For a CUDA build on a GPU node:
```bash
module purge
module load gcc
module load openmpi
module load cuda
```

The exact module names can vary slightly by cluster and semester. If one module name is unavailable, use the closest available GCC / OpenMPI / CUDA module on your CHPC system.

## Compile commands

### Serial CPU
```bash
g++ -std=c++11 -O2 serial_vorticity.cpp -o serial_vorticity
```

### OpenMP CPU
```bash
g++ -std=c++11 -O2 -fopenmp openmp_vorticity.cpp -o openmp_vorticity
```

### MPI CPU
```bash
mpic++ -std=c++11 -O2 mpi_vorticity.cpp -o mpi_vorticity
```

### CUDA GPU
```bash
nvcc -O2 -std=c++11 cuda_vorticity.cu -o cuda_vorticity
```

### MPI + CUDA GPU
```bash
nvcc -ccbin mpic++ -O2 -std=c++11 mpi_cuda_vorticity.cu -o mpi_cuda_vorticity
```

## Run commands

In all examples below, replace the input path if needed.

### Serial CPU
```bash
./serial_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix serial --width 1300 --height 600
```

### OpenMP CPU
```bash
export OMP_NUM_THREADS=8
./openmp_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix omp --width 1300 --height 600
```

### MPI CPU
```bash
mpirun -np 4 ./mpi_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix mpi_cpu --width 1300 --height 600
```

### CUDA GPU
```bash
./cuda_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix cuda --width 1300 --height 600 --block-x 16 --block-y 16
```

### MPI + CUDA GPU
Run one rank per GPU.
```bash
mpirun -np 2 ./mpi_cuda_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix mpi_cuda --width 1300 --height 600 --block-x 16 --block-y 16
```

## Simple CHPC batch examples

### CPU batch job
```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=notchpeak-freecycle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00

module purge
module load gcc
module load openmpi

export OMP_NUM_THREADS=8
./openmp_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix omp_run --width 1300 --height 600
```

### GPU batch job
```bash
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=notchpeak-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

module purge
module load gcc
module load openmpi
module load cuda

./cuda_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix cuda_run --width 1300 --height 600 --block-x 16 --block-y 16
```

## Description of approach used for each implementation

### 1. Serial CPU
- Read the full field into memory.
- Compute magnitude from `(u, v)` at every cell.
- Compute vorticity at every cell with the provided starter function.
- Write scalar raw outputs.

### 2. Parallel shared-memory CPU
- Reuse the same raw input and output logic.
- Parallelize the outer loops over `(x, y)` with OpenMP.
- Each thread writes to unique output cells, so no reductions are required.

### 3. Parallel CUDA GPU
- Copy the field to the GPU.
- Launch one CUDA thread per cell.
- Each thread computes magnitude and vorticity for its own cell.
- Copy scalar results back to the CPU and write them to files.
- No CUDA shared memory is used.

### 4. Distributed-memory CPU
- Split the grid by rows across MPI ranks.
- Scatter the local row blocks to each rank.
- Exchange one halo row above and below with neighboring ranks.
- Compute local magnitude and local vorticity.
- Gather the final scalar results back to rank 0 and write them.

### 5. Distributed-memory GPU
- Split the grid by rows across MPI ranks.
- Exchange halo rows with MPI on the CPU side.
- Copy the local extended block to the GPU for each rank.
- Launch one CUDA thread per local cell.
- Gather the local scalar results back to rank 0.

## Scaling study scripts (CHPC)

The project includes ready-to-run SLURM scripts in `slurm/`:
- `slurm/cpu_scaling.slurm` - serial vs OpenMP thread scaling (`1, 2, 4, 8, 16`)
- `slurm/cuda_blocks.slurm` - CUDA block-size comparison (`8x8, 16x16, 32x8, 32x16`)
- `slurm/mpi_cpu_scaling.slurm` - MPI CPU scaling (`2` and `4` ranks/nodes)
- `slurm/mpi_cuda_scaling.slurm` - MPI+CUDA scaling (`2` and `4` ranks/nodes, one GPU per rank)

Before submitting jobs:
- Set your CHPC account in each script (`#SBATCH --account=YOUR_ACCOUNT`).
- Adjust partitions if your environment uses different names.
- Confirm the input path points to your raw file.
- For MPI jobs on this CHPC setup, use `mpirun` inside the allocation (not direct `srun` launch with OpenMPI).

Submit from the project root so outputs and relative paths resolve correctly:
```bash
cd ~/Project/hpc_vorticity_project

sbatch -A usucs6030 -p notchpeak-freecycle --chdir="$PWD" slurm/cpu_scaling.slurm
sbatch -A notchpeak-gpu -p notchpeak-gpu --chdir="$PWD" slurm/cuda_blocks.slurm
sbatch -A usucs6030 -p notchpeak-freecycle --chdir="$PWD" slurm/mpi_cpu_scaling.slurm
sbatch -A notchpeak-gpu -p notchpeak-gpu --chdir="$PWD" slurm/mpi_cuda_scaling.slurm
```

If your account/partition names differ, check:
```bash
sinfo -s
sacctmgr -n show assoc where user=$USER format=Account%30,Partition%30,QOS%30
```

Each script writes CSV output in `results/`:
- `results/cpu_scaling_<jobid>.csv`
- `results/cuda_blocks_<jobid>.csv`
- `results/mpi_cpu_scaling_<jobid>.csv`
- `results/mpi_cuda_scaling_<jobid>.csv`

Timing interpretation:
- CPU CSV files report runtime in seconds.
- CUDA CSV files report kernel runtime in milliseconds (kernel-only, not full end-to-end runtime).

Use `templates/scaling_results_template.csv` as a report table template.

## Validation

Use the serial vorticity file as the reference:
```bash
python3 validate.py serial_vorticity.raw omp_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw mpi_cpu_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw cuda_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw mpi_cuda_vorticity.raw 1300 600
```

## Visualization

Generate PNG images from the raw outputs:
```bash
python3 visualize.py serial_vorticity.raw serial_magnitude.raw 1300 600 serial
```

## Work division for this submission

Author: Sam Willden

Completed by this author:
- serial CPU implementation
- OpenMP CPU implementation
- CUDA GPU implementation
- MPI CPU implementation
- MPI + CUDA implementation
- validation script usage and correctness checks
- CHPC job scripts and scaling experiments
- visualization output generation
- README and results documentation

## Notes

- The CPU versions reuse the exact starter vorticity function.
- The GPU versions mirror the same indexing and finite-difference logic.
- All outputs are written as float32 raw scalar fields.
- On this CHPC OpenMPI build, MPI jobs should be launched with `mpirun` inside SLURM allocations.
- If you reuse any external code in the future, cite it in your report and source comments.
