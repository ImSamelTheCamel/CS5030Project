#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

#include "vorticity_starter.hpp"
#include "common.hpp"

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        return 1;
    }

    std::vector<float> field;
    if (!read_raw_field(args.input_file, field, args.width, args.height)) {
        return 1;
    }

    std::vector<float> magnitude(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));
    std::vector<float> vort(static_cast<size_t>(args.width) * static_cast<size_t>(args.height));

    double t0 = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < args.height; y++) {
        for (int x = 0; x < args.width; x++) {
            int idx = y * args.width + x;
            int fidx = idx * 2;
            float u = field[fidx];
            float v = field[fidx + 1];
            magnitude[idx] = std::sqrt(u * u + v * v);
            vort[idx] = vorticity(x, y, args.width, args.height, field.data());
        }
    }

    double t1 = omp_get_wtime();

    std::string vort_name = args.output_prefix + "_vorticity.raw";
    std::string mag_name = args.output_prefix + "_magnitude.raw";

    if (!write_raw_scalar(vort_name, vort) || !write_raw_scalar(mag_name, magnitude)) {
        return 1;
    }

    std::cout << "OpenMP runtime (s): " << (t1 - t0) << "\n";
    std::cout << "Threads used: " << omp_get_max_threads() << "\n";
    std::cout << "Wrote " << vort_name << " and " << mag_name << "\n";
    return 0;
}
