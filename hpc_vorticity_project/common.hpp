#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdlib>

struct Args {
    std::string input_file = "cyl2d_1300x600_float32[2].raw";
    std::string output_prefix = "out";
    int width = 1300;
    int height = 600;
    int block_x = 16;
    int block_y = 16;
};

inline bool parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--input" && i + 1 < argc) {
            args.input_file = argv[++i];
        } else if (s == "--output-prefix" && i + 1 < argc) {
            args.output_prefix = argv[++i];
        } else if (s == "--width" && i + 1 < argc) {
            args.width = std::atoi(argv[++i]);
        } else if (s == "--height" && i + 1 < argc) {
            args.height = std::atoi(argv[++i]);
        } else if (s == "--block-x" && i + 1 < argc) {
            args.block_x = std::atoi(argv[++i]);
        } else if (s == "--block-y" && i + 1 < argc) {
            args.block_y = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << s << "\n";
            return false;
        }
    }
    return true;
}

inline bool read_raw_field(const std::string& path, std::vector<float>& field, int width, int height) {
    std::ifstream fin(path.c_str(), std::ios::binary);
    if (!fin) {
        std::cerr << "Could not open input file: " << path << "\n";
        return false;
    }
    size_t total = static_cast<size_t>(width) * static_cast<size_t>(height) * 2ull;
    field.resize(total);
    fin.read(reinterpret_cast<char*>(field.data()), static_cast<std::streamsize>(total * sizeof(float)));
    if (!fin) {
        std::cerr << "Could not read expected number of bytes from: " << path << "\n";
        return false;
    }
    return true;
}

inline bool write_raw_scalar(const std::string& path, const std::vector<float>& data) {
    std::ofstream fout(path.c_str(), std::ios::binary);
    if (!fout) {
        std::cerr << "Could not open output file: " << path << "\n";
        return false;
    }
    fout.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return static_cast<bool>(fout);
}

inline void compute_magnitude_serial(const std::vector<float>& field, std::vector<float>& magnitude, int width, int height) {
    magnitude.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int fidx = idx * 2;
            float u = field[fidx];
            float v = field[fidx + 1];
            magnitude[idx] = std::sqrt(u * u + v * v);
        }
    }
}

inline double now_seconds() {
    using clock = std::chrono::high_resolution_clock;
    auto t = clock::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

#endif
