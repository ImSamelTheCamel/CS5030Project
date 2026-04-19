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