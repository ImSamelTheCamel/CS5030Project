# Approach Used for Each Implementation

## Serial Implementation
The serial version is the simplest version of the program and is used as the baseline for correctness. In this approach, the program reads the entire 2D vector field into memory and processes one cell at a time using a regular loop. For each cell, it computes the vorticity using the given formula and then applies the thresholds to classify the result as $-1$, $0$, or $+1$. After all cells are processed, the results are written to an output file. This version is important because all other implementations are compared against it to make sure they are correct.

## Parallel Shared-Memory CPU Implementation
The shared-memory version uses OpenMP to speed up the computation by using multiple CPU threads. Instead of one loop running on a single core, the work is split across several threads. Each thread handles different cells in the grid at the same time. Since each cell is independent, this works well and does not require complicated synchronization. The overall logic is the same as the serial version, but it runs faster because it uses multiple cores on the CPU.

## Parallel CUDA GPU Implementation
The CUDA version runs the computation on a GPU. First, the input data is copied from the CPU to the GPU. Then, a kernel is launched where each GPU thread is responsible for computing the result for one cell. Each thread calculates the vorticity and applies the threshold to produce $-1$, $0$, or $+1$. After all threads finish, the results are copied back to the CPU and written to a file. This version does not use shared memory, which keeps it simple, and performance is improved by adjusting the block size.

## Distributed-Memory CPU Implementation
The distributed CPU version uses MPI to split the work across multiple processes. Each process is responsible for a portion of the rows in the grid. Because the vorticity calculation depends on neighboring cells, each process also needs data from the rows just above and below its assigned section. This is handled using a halo exchange between neighboring processes. After receiving the needed data, each process computes its portion of the result. Finally, all results are sent back to the main process, which writes the output file. This version allows the program to run across multiple machines or nodes.

## Distributed-Memory GPU Implementation
The distributed GPU version combines MPI and CUDA. Like the MPI CPU version, the data is split across multiple processes. Each process exchanges halo rows with its neighbors to get the correct boundary data. Then, instead of computing on the CPU, each process sends its portion of the data to a GPU and runs a CUDA kernel. Each GPU thread computes one cell, just like in the single-GPU version. After the computation, results are copied back to the CPU and gathered into one final output. This version uses both multiple machines and GPUs, making it the most powerful and scalable version.

## Summary
All implementations use the same basic idea for computing vorticity and applying thresholds. The main difference is how the work is split and what hardware is used. The serial version runs on one core, OpenMP uses multiple CPU cores, CUDA uses a GPU, MPI uses multiple processes across nodes, and MPI+CUDA combines both distributed computing and GPUs for the best performance.