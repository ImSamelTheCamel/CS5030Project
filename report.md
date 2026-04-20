# Approach Used for Each Implementation

## Serial Implementation
The serial version is the most basic version of the program and is mainly used to make sure everything works correctly. In this approach, the program reads in the entire 2D vector field and processes one cell at a time using a simple loop. For each cell, it calculates the vorticity using the formula, then applies the thresholds to decide if the result should be $-1$, $0$, or $+1$. Once all the cells are processed, the results are written to an output file. This version is important because we use it to check that the other, faster versions are still giving the correct results.

## Parallel Shared-Memory CPU Implementation
This version uses OpenMP to speed things up by using multiple CPU cores. Instead of doing all the work on one core, the loop is split so multiple threads can work at the same time. Each thread handles different parts of the grid. Since each cell can be computed independently, we don’t have to worry much about threads interfering with each other. The logic is basically the same as the serial version, but it runs faster because it uses more of the CPU.

## Parallel CUDA GPU Implementation
In this version, the computation is done on a GPU instead of the CPU. First, the input data is copied from the CPU to the GPU. Then, a CUDA kernel is launched where each thread handles one cell in the grid. Each thread computes the vorticity and applies the threshold to get $-1$, $0$, or $+1$. After all the threads finish, the results are copied back to the CPU and written to a file. This version doesn’t use shared memory, which keeps it simpler, and performance can be improved by choosing a good block size.

## Distributed-Memory CPU Implementation
This version uses MPI to split the work across multiple processes, which can run on different machines. Each process is responsible for a chunk of rows from the grid. Since the vorticity calculation needs neighboring cells, each process also needs a little extra data from the rows above and below its section. This is handled by sending and receiving data (called a halo exchange) between processes. After getting the needed data, each process does its calculations and then sends the results back to the main process, which writes the final output file.

## Distributed-Memory GPU Implementation
This version combines MPI and CUDA. Like the MPI CPU version, the grid is split across multiple processes, and each process exchanges boundary (halo) data with its neighbors. But instead of doing the calculations on the CPU, each process sends its data to a GPU and runs a CUDA kernel. Each GPU thread computes one cell, just like before. After the computation, the results are copied back to the CPU and combined into the final output. This version uses both multiple machines and GPUs, making it the fastest and most scalable option.

## Summary
All of these versions follow the same basic idea for computing vorticity and applying thresholds. The main difference is how the work is divided and what hardware is used. The serial version runs on a single core, OpenMP uses multiple CPU cores, CUDA uses a GPU, MPI spreads the work across multiple machines, and MPI + CUDA combines everything for the best performance.

# Build and Run Instructions (CHPC)

## Environment Setup
For CPU-only builds:
```bash
module purge
module load gcc
module load openmpi
```

For CUDA builds:
```bash
module purge
module load gcc
module load openmpi
module load cuda
```

## Compile Commands
```bash
g++ -std=c++11 -O2 serial_vorticity.cpp -o serial_vorticity
g++ -std=c++11 -O2 -fopenmp openmp_vorticity.cpp -o openmp_vorticity
mpic++ -std=c++11 -O2 mpi_vorticity.cpp -o mpi_vorticity
nvcc -O2 -std=c++11 cuda_vorticity.cu -o cuda_vorticity
nvcc -ccbin mpic++ -O2 -std=c++11 mpi_cuda_vorticity.cu -o mpi_cuda_vorticity
```

## Example Run Commands
```bash
./serial_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix serial --width 1300 --height 600

export OMP_NUM_THREADS=8
./openmp_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix omp --width 1300 --height 600

mpirun -np 4 ./mpi_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix mpi_cpu --width 1300 --height 600

./cuda_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix cuda --width 1300 --height 600 --block-x 16 --block-y 16

mpirun -np 2 ./mpi_cuda_vorticity --input cyl2d_1300x600_float32[2].raw --output-prefix mpi_cuda --width 1300 --height 600 --block-x 16 --block-y 16
```

CHPC launch note: on this OpenMPI setup, the working method was `sbatch` submission and `mpirun` inside the allocation.

# Scaling Experiments and Results

System: University of Utah CHPC (notchpeak)  
Date: 2026-04-18  
Dataset: `cyl2d_1300x600_float32[2].raw`  
Grid: `1300 x 600`  
Summary values below are medians of 5 runs.

## Serial Baseline
- Serial median runtime: `0.00696158 s`

## 1 vs 2: Serial vs OpenMP
- OpenMP 1 thread: `0.00610536 s` (speedup `1.14x`)
- OpenMP 2 threads: `0.00294480 s` (speedup `2.36x`)
- OpenMP 4 threads: `0.00325876 s` (speedup `2.14x`)
- OpenMP 8 threads: `0.01579860 s` (speedup `0.44x`)
- OpenMP 16 threads: `0.02156770 s` (speedup `0.32x`)

## 3: CUDA Block-Size Experiment (Kernel-Only)
- `8x8`: `0.041440 ms`
- `16x16`: `0.036384 ms`
- `32x8`: `0.038592 ms`
- `32x16`: `0.035936 ms` (best median)

## 4 vs 5: Distributed CPU vs Distributed GPU
MPI CPU (runtime in seconds):
- 2 nodes / 2 ranks: `0.00323051 s` (speedup vs serial `2.15x`)
- 4 nodes / 4 ranks: `0.00161524 s` (speedup vs serial `4.31x`)

MPI + CUDA (local kernel runtime in milliseconds, block `16x16`):
- 2 nodes / 2 ranks: `0.024672 ms`
- 4 nodes / 4 ranks: `0.019936 ms`

Timing note: CUDA and MPI+CUDA numbers above are kernel-only timing, not full end-to-end runtime.

Primary result summary source: `hpc_vorticity_project/results.txt`.

# Validation Against Serial Output

- Validation script used: `hpc_vorticity_project/validate.py`
- Compared OpenMP, MPI CPU, CUDA, and MPI+CUDA outputs against serial output.
- Validation status: all checks passed.
- Maximum absolute difference reported: `0.0`

Example validation commands:
```bash
python3 validate.py serial_vorticity.raw omp_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw mpi_cpu_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw cuda_vorticity.raw 1300 600
python3 validate.py serial_vorticity.raw mpi_cuda_vorticity.raw 1300 600
```

# Code Reuse Across Implementations

- Shared argument parsing, raw I/O, timing, and utility logic are centralized in `hpc_vorticity_project/common.hpp`.
- CPU implementations reuse the supplied starter vorticity function without modification (`hpc_vorticity_project/vorticity_starter.hpp`).
- GPU kernels mirror the same finite-difference/indexing logic used in CPU versions to preserve correctness.
- This reuse pattern reduced duplication and helped keep behavior consistent across serial, OpenMP, MPI CPU, CUDA, and MPI+CUDA implementations.

# Visualization Output

- Visualization utility: `hpc_vorticity_project/visualize.py`
- Generated artifacts:
  - `hpc_vorticity_project/serial_vorticity.png`
  - `hpc_vorticity_project/serial_magnitude.png`

Example command:
```bash
python3 visualize.py serial_vorticity.raw serial_magnitude.raw 1300 600 serial
```

# Responsibility Split

This submission is single-author.

Author: Sam Willden

Completed by this author:
- serial CPU implementation
- OpenMP CPU implementation
- CUDA GPU implementation
- MPI CPU implementation
- MPI + CUDA implementation
- validation and correctness checks
- CHPC SLURM scripts and scaling experiments
- visualization output generation
- report and project documentation
