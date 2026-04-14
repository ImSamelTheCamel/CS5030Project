#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "common.hpp"

__device__ float device_vorticity(int x, int y, int width, int height, const float* f){
    float dx = 0.01f;
    float dy = 0.01f;

    unsigned int idx = y * width + x;

    int start_x = (x == 0) ? 0 : x - 1;
    int end_x = (x == width - 1) ? x : x + 1;

    int start_y = (y == 0) ? 0 : y - 1;
    int end_y = (y == height - 1) ? y : y + 1;

    unsigned int duidx = (start_y * width + end_x) * 2;
    unsigned int dvidx = (end_y * width + start_x) * 2;

    double fdu0 = f[duidx];
    double fdu1 = f[duidx + 1];
    double fdv0 = f[dvidx];
    double fdv1 = f[dvidx + 1];
    double vec00 = f[idx * 2];
    double vec01 = f[idx * 2 + 1];

    float duy = (fdu1 - vec01) / (dx * (end_x - start_x));
    float dvx = (fdv0 - vec00) / (dy * (end_y - start_y));

    return duy - dvx;
}

__global__ void compute_kernel(const float* field, float* vort, float* magnitude, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    int fidx = idx * 2;
    float u = field[fidx];
    float v = field[fidx + 1];

    magnitude[idx] = sqrtf(u * u + v * v);
    vort[idx] = device_vorticity(x, y, width, height, field);
}

static bool check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    std::vector<float> field;
    if (!read_raw_field(args.input_file, field, args.width, args.height)) {
        return 1;
    }

    size_t field_bytes = static_cast<size_t>(args.width) * static_cast<size_t>(args.height) * 2ull * sizeof(float);
    size_t scalar_bytes = static_cast<size_t>(args.width) * static_cast<size_t>(args.height) * sizeof(float);

    float* d_field = nullptr;
    float* d_vort = nullptr;
    float* d_mag = nullptr;

    if (!check_cuda(cudaMalloc(&d_field, field_bytes), "cudaMalloc field")) return 1;
    if (!check_cuda(cudaMalloc(&d_vort, scalar_bytes), "cudaMalloc vort")) return 1;
    if (!check_cuda(cudaMalloc(&d_mag, scalar_bytes), "cudaMalloc mag")) return 1;
    if (!check_cuda(cudaMemcpy(d_field, field.data(), field_bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D")) return 1;

    dim3 block(args.block_x, args.block_y);
    dim3 grid((args.width + block.x - 1) / block.x, (args.height + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    compute_kernel<<<grid, block>>>(d_field, d_vort, d_mag, args.width, args.height);
    if (!check_cuda(cudaGetLastError(), "kernel launch")) return 1;
    if (!check_cuda(cudaDeviceSynchronize(), "kernel sync")) return 1;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<float> vort(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));
    std::vector<float> magnitude(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));

    if (!check_cuda(cudaMemcpy(vort.data(), d_vort, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy vort D2H")) return 1;
    if (!check_cuda(cudaMemcpy(magnitude.data(), d_mag, scalar_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy mag D2H")) return 1;

    std::string vort_name = args.output_prefix + "_vorticity.raw";
    std::string mag_name = args.output_prefix + "_magnitude.raw";
    if (!write_raw_scalar(vort_name, vort) || !write_raw_scalar(mag_name, magnitude)) {
        return 1;
    }

    std::cout << "CUDA kernel runtime (ms): " << ms << "\n";
    std::cout << "Block size: " << args.block_x << "x" << args.block_y << "\n";
    std::cout << "Wrote " << vort_name << " and " << mag_name << "\n";

    cudaFree(d_field);
    cudaFree(d_vort);
    cudaFree(d_mag);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
