#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>

#include "common.hpp"

static void local_row_range(int rank, int size, int global_height, int& start_row, int& local_rows) {
    int base = global_height / size;
    int rem = global_height % size;
    local_rows = base + (rank < rem ? 1 : 0);
    start_row = rank * base + (rank < rem ? rank : rem);
}

__device__ float local_device_vorticity(int x,
                                        int local_y,
                                        int width,
                                        int global_y,
                                        int global_height,
                                        const float* f) {
    float dx = 0.01f;
    float dy = 0.01f;

    int start_x = (x == 0) ? 0 : x - 1;
    int end_x = (x == width - 1) ? x : x + 1;

    int start_y = (global_y == 0) ? local_y : local_y - 1;
    int end_y = (global_y == global_height - 1) ? local_y : local_y + 1;

    unsigned int idx = local_y * width + x;
    unsigned int duidx = (start_y * width + end_x) * 2;
    unsigned int dvidx = (end_y * width + start_x) * 2;

    double fdu1 = f[duidx + 1];
    double fdv0 = f[dvidx];
    double vec00 = f[idx * 2];
    double vec01 = f[idx * 2 + 1];

    float duy = (fdu1 - vec01) / (dx * (end_x - start_x));
    float dvx = (fdv0 - vec00) / (dy * (end_y - start_y));
    return duy - dvx;
}

__global__ void local_kernel(const float* field_ext,
                             float* vort,
                             float* mag,
                             int width,
                             int local_rows,
                             int start_row,
                             int global_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || ly >= local_rows) {
        return;
    }

    int local_idx = ly * width + x;
    int ext_y = ly + 1;
    int ext_idx = ext_y * width + x;
    int fidx = ext_idx * 2;

    float u = field_ext[fidx];
    float v = field_ext[fidx + 1];
    mag[local_idx] = sqrtf(u * u + v * v);

    int global_y = start_row + ly;
    vort[local_idx] = local_device_vorticity(x, ext_y, width, global_y, global_height, field_ext);
}

static bool check_cuda(cudaError_t err, const char* msg, int rank) {
    if (err != cudaSuccess) {
        std::cerr << "Rank " << rank << " " << msg << ": " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Args args;
    if (!parse_args(argc, argv, args)) {
        MPI_Finalize();
        return 1;
    }

    int gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    if (gpu_count < 1) {
        if (rank == 0) {
            std::cerr << "No CUDA devices found.\n";
        }
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(rank % gpu_count);

    int start_row = 0;
    int local_rows = 0;
    local_row_range(rank, size, args.height, start_row, local_rows);

    std::vector<float> global_field;
    std::vector<int> sendcounts(size), displs(size), scalar_counts(size), scalar_displs(size);
    if (rank == 0) {
        if (!read_raw_field(args.input_file, global_field, args.width, args.height)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int r = 0; r < size; r++) {
            int srow = 0;
            int lrows = 0;
            local_row_range(r, size, args.height, srow, lrows);
            sendcounts[r] = lrows * args.width * 2;
            displs[r] = srow * args.width * 2;
            scalar_counts[r] = lrows * args.width;
            scalar_displs[r] = srow * args.width;
        }
    }

    std::vector<float> local_field_ext(static_cast<size_t>(local_rows + 2) * static_cast<size_t>(args.width) * 2ull, 0.0f);
    std::vector<float> local_compact(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width) * 2ull);

    MPI_Scatterv(rank == 0 ? global_field.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_FLOAT,
                 local_compact.data(), local_rows * args.width * 2, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    for (int r = 0; r < local_rows; r++) {
        std::copy(local_compact.begin() + static_cast<size_t>(r) * args.width * 2,
                  local_compact.begin() + static_cast<size_t>(r + 1) * args.width * 2,
                  local_field_ext.begin() + static_cast<size_t>(r + 1) * args.width * 2);
    }

    MPI_Request reqs[4];
    int req_count = 0;
    if (rank > 0) {
        MPI_Irecv(local_field_ext.data(), args.width * 2, MPI_FLOAT, rank - 1, 200, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_field_ext.data() + static_cast<size_t>(1) * args.width * 2, args.width * 2, MPI_FLOAT, rank - 1, 201, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(local_field_ext.data() + static_cast<size_t>(local_rows + 1) * args.width * 2, args.width * 2, MPI_FLOAT, rank + 1, 201, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_field_ext.data() + static_cast<size_t>(local_rows) * args.width * 2, args.width * 2, MPI_FLOAT, rank + 1, 200, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    size_t ext_bytes = static_cast<size_t>(local_rows + 2) * static_cast<size_t>(args.width) * 2ull * sizeof(float);
    size_t scalar_bytes = static_cast<size_t>(local_rows) * static_cast<size_t>(args.width) * sizeof(float);

    float* d_field_ext = nullptr;
    float* d_vort = nullptr;
    float* d_mag = nullptr;

    if (!check_cuda(cudaMalloc(&d_field_ext, ext_bytes), "cudaMalloc field", rank)) return 1;
    if (!check_cuda(cudaMalloc(&d_vort, scalar_bytes), "cudaMalloc vort", rank)) return 1;
    if (!check_cuda(cudaMalloc(&d_mag, scalar_bytes), "cudaMalloc mag", rank)) return 1;
    if (!check_cuda(cudaMemcpy(d_field_ext, local_field_ext.data(), ext_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D", rank)) return 1;

    dim3 block(args.block_x, args.block_y);
    dim3 grid((args.width + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    local_kernel<<<grid, block>>>(d_field_ext, d_vort, d_mag, args.width, local_rows, start_row, args.height);
    if (!check_cuda(cudaGetLastError(), "kernel launch", rank)) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "kernel sync", rank)) return 1;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<float> local_vort(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width));
    std::vector<float> local_mag(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width));

    if (!check_cuda(cudaMemcpy(local_vort.data(), d_vort, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy vort D2H", rank)) return 1;
    if (!check_cuda(cudaMemcpy(local_mag.data(), d_mag, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy mag D2H", rank)) return 1;

    std::vector<float> global_vort;
    std::vector<float> global_mag;
    if (rank == 0) {
        global_vort.resize(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));
        global_mag.resize(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));
    }

    MPI_Gatherv(local_vort.data(), local_rows * args.width, MPI_FLOAT,
                rank == 0 ? global_vort.data() : nullptr,
                scalar_counts.data(), scalar_displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(local_mag.data(), local_rows * args.width, MPI_FLOAT,
                rank == 0 ? global_mag.data() : nullptr,
                scalar_counts.data(), scalar_displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::string vort_name = args.output_prefix + "_vorticity.raw";
        std::string mag_name = args.output_prefix + "_magnitude.raw";
        write_raw_scalar(vort_name, global_vort);
        write_raw_scalar(mag_name, global_mag);
        std::cout << "MPI CUDA local kernel runtime (ms): " << ms << "\n";
        std::cout << "MPI ranks / GPUs: " << size << "\n";
        std::cout << "Block size: " << args.block_x << "x" << args.block_y << "\n";
        std::cout << "Wrote " << vort_name << " and " << mag_name << "\n";
    }

    cudaFree(d_field_ext);
    cudaFree(d_vort);
    cudaFree(d_mag);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    MPI_Finalize();
    return 0;
}
