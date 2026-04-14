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
#SBATCH --partition=kingspeak
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
#SBATCH --partition=kingspeak-gpu
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

## Scaling study plan

### Compare 1 vs 2
Run:
- `serial_vorticity`
- `openmp_vorticity`
- `cuda_vorticity`
- `mpi_vorticity`
- `mpi_cuda_vorticity`

Record runtime and speedup relative to serial.

### Compare 3
For OpenMP:
- test thread counts such as 1, 2, 4, 8

For CUDA:
- test block sizes such as `8x8`, `16x16`, `32x8`, `32x16`

### Compare 4 vs 5
Use 2 to 4 nodes or 2 to 4 ranks with one GPU per rank where available.
- MPI CPU: compare `-np 2` and `-np 4`
- MPI CUDA: compare `-np 2` and `-np 4`

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

## Work division example for a 2-person team

Person 1:
- serial CPU
- OpenMP CPU
- MPI CPU
- validation scripts and CPU scaling plots

Person 2:
- CUDA GPU
- MPI + CUDA GPU
- visualization scripts
- GPU block-size experiments and report figures

Shared:
- correctness checks
- batch scripts
- final report and presentation

## Notes

- The CPU versions reuse the exact starter vorticity function.
- The GPU versions mirror the same indexing and finite-difference logic.
- All outputs are written as float32 raw scalar fields.
- If you reuse any external code in the future, cite it in your report and source comments.
