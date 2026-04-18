# Submission Requirement Status

This file maps project artifacts to the assignment requirements.

## 1) Code + CHPC build/run instructions
Status: Complete
- Code for all implementations is in this repository.
- CHPC build/run instructions are in `README.md`.
- Working CHPC launch method documented (submit with `sbatch`, run MPI via `mpirun` in allocation).

## 2) Approach description for each implementation
Status: Complete
- Serial CPU: documented in `README.md`.
- Parallel shared-memory CPU (OpenMP): documented in `README.md`.
- Parallel CUDA GPU (no shared memory): documented in `README.md`.
- Distributed-memory CPU (MPI): documented in `README.md`.
- Distributed-memory GPU (MPI + CUDA): documented in `README.md`.

## 3) Scaling experiments
Status: Complete
- 1 vs 2 (serial vs OpenMP thread scaling): `results/cpu_scaling_12182521.csv`
- 3 (CUDA block-size experiment): `results/cuda_blocks_12182519.csv`
- 4 vs 5 (MPI CPU vs MPI+CUDA on 2 and 4 nodes/ranks):
  - `results/mpi_cpu_scaling_12183703.csv`
  - `results/mpi_cuda_scaling_12183704.csv`
- Summary statistics are recorded in `results.txt`.

## 4) Validation function against serial output
Status: Complete
- Validation script: `validate.py`
- Parallel outputs validated against serial output.

## 5) Code reuse across implementations
Status: Complete
- Shared parsing/I/O/timing/common logic in `common.hpp`.
- Common numerical logic is reused/mirrored consistently across implementations.

## 6) Visualization output
Status: Complete
- Visualization script: `visualize.py`
- Generated example outputs:
  - `serial_vorticity.png`
  - `serial_magnitude.png`

## 7) Responsibility split
Status: Complete for this submission
- This submission is a single-author implementation.
- Author: Sam Willden.
- Responsibilities: all implementations, CHPC scripts/runs, validation, scaling analysis, visualization, and documentation.
