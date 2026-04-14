#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <mpi.h>

#include "common.hpp"
#include "vorticity_starter.hpp"

static void local_row_range(int rank, int size, int global_height, int& start_row, int& local_rows) {
    int base = global_height / size;
    int rem = global_height % size;
    local_rows = base + (rank < rem ? 1 : 0);
    start_row = rank * base + (rank < rem ? rank : rem);
}

static float local_vorticity_with_halo(int x,
                                       int local_y,
                                       int width,
                                       int local_rows,
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

    Vec2d fdu(f[duidx], f[duidx + 1]);
    Vec2d fdv(f[dvidx], f[dvidx + 1]);
    Vec2d vec0(f[idx * 2], f[idx * 2 + 1]);

    float duy = (fdu.second - vec0.second) / (dx * (end_x - start_x));
    float dvx = (fdv.first - vec0.first) / (dy * (end_y - start_y));

    return duy - dvx;
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

    std::vector<float> local_field(static_cast<size_t>(local_rows + 2) * static_cast<size_t>(args.width) * 2ull, 0.0f);
    std::vector<float> local_compact(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width) * 2ull);

    MPI_Scatterv(rank == 0 ? global_field.data() : nullptr,
                 sendcounts.data(), displs.data(), MPI_FLOAT,
                 local_compact.data(), local_rows * args.width * 2, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    for (int r = 0; r < local_rows; r++) {
        std::copy(local_compact.begin() + static_cast<size_t>(r) * args.width * 2,
                  local_compact.begin() + static_cast<size_t>(r + 1) * args.width * 2,
                  local_field.begin() + static_cast<size_t>(r + 1) * args.width * 2);
    }

    MPI_Request reqs[4];
    int req_count = 0;
    if (rank > 0) {
        MPI_Irecv(local_field.data(), args.width * 2, MPI_FLOAT, rank - 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_field.data() + static_cast<size_t>(1) * args.width * 2, args.width * 2, MPI_FLOAT, rank - 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (rank < size - 1) {
        MPI_Irecv(local_field.data() + static_cast<size_t>(local_rows + 1) * args.width * 2, args.width * 2, MPI_FLOAT, rank + 1, 101, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Isend(local_field.data() + static_cast<size_t>(local_rows) * args.width * 2, args.width * 2, MPI_FLOAT, rank + 1, 100, MPI_COMM_WORLD, &reqs[req_count++]);
    }
    if (req_count > 0) {
        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    std::vector<float> local_vort(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width));
    std::vector<float> local_mag(static_cast<size_t>(local_rows) * static_cast<size_t>(args.width));

    double t0 = MPI_Wtime();
    for (int ly = 0; ly < local_rows; ly++) {
        for (int x = 0; x < args.width; x++) {
            int global_y = start_row + ly;
            int local_idx = ly * args.width + x;
            int compact_idx = local_idx * 2;
            float u = local_compact[compact_idx];
            float v = local_compact[compact_idx + 1];
            local_mag[local_idx] = std::sqrt(u * u + v * v);
            local_vort[local_idx] = local_vorticity_with_halo(x, ly + 1, args.width, local_rows, global_y, args.height, local_field.data());
        }
    }
    double t1 = MPI_Wtime();

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
        std::cout << "MPI CPU runtime (s): " << (t1 - t0) << "\n";
        std::cout << "MPI ranks: " << size << "\n";
        std::cout << "Wrote " << vort_name << " and " << mag_name << "\n";
    }

    MPI_Finalize();
    return 0;
}
