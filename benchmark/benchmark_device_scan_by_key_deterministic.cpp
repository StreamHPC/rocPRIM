// MIT License
//
// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "benchmark_device_scan_by_key.parallel.hpp"
#include "benchmark_utils.hpp"
// CmdParser
#include "cmdparser.hpp"

// Google Benchmark
#include <benchmark/benchmark.h>

// HIP API
#include <hip/hip_runtime.h>

#include <string>

#include <cstddef>

#ifndef DEFAULT_N
const size_t DEFAULT_N = 1024 * 1024 * 32;
#endif

#define CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, MAX_SEGMENT_LENGTH) \
    {                                                                 \
        const device_scan_by_key_benchmark<EXCL,                      \
                                           int,                       \
                                           T,                         \
                                           SCAN_OP,                   \
                                           rocprim::equal_to<int>,    \
                                           MAX_SEGMENT_LENGTH,        \
                                           true>                      \
            instance;                                                 \
        REGISTER_BENCHMARK(benchmarks, size, seed, stream, instance); \
    }

#define CREATE_EXCL_INCL_BENCHMARK(EXCL, T, SCAN_OP) \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 1)     \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 16)    \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 256)   \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 4096)  \
    CREATE_BY_KEY_BENCHMARK(EXCL, T, SCAN_OP, 65536)

#define CREATE_BENCHMARK(T, SCAN_OP)              \
    CREATE_EXCL_INCL_BENCHMARK(false, T, SCAN_OP) \
    CREATE_EXCL_INCL_BENCHMARK(true, T, SCAN_OP)

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_N, "number of values");
    parser.set_optional<int>("trials", "trials", -1, "number of iterations");
    parser.set_optional<std::string>("name_format",
                                     "name_format",
                                     "human",
                                     "either: json,human,txt");
    parser.set_optional<std::string>("seed", "seed", "random", get_seed_message());
    parser.run_and_exit_if_error();

    // Parse argv
    benchmark::Initialize(&argc, argv);
    const size_t size   = parser.get<size_t>("size");
    const int    trials = parser.get<int>("trials");
    bench_naming::set_format(parser.get<std::string>("name_format"));
    const std::string  seed_type = parser.get<std::string>("seed");
    const managed_seed seed(seed_type);

    // HIP
    hipStream_t stream = 0; // default

    // Benchmark info
    add_common_benchmark_info();
    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("seed", seed_type);

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
    using custom_float2                                     = custom_type<float, float>;
    using custom_double2                                    = custom_type<double, double>;

    CREATE_BENCHMARK(int, rocprim::plus<int>)
    CREATE_BENCHMARK(float, rocprim::plus<float>)
    CREATE_BENCHMARK(double, rocprim::plus<double>)
    CREATE_BENCHMARK(long long, rocprim::plus<long long>)
    CREATE_BENCHMARK(float2, rocprim::plus<float2>)
    CREATE_BENCHMARK(custom_float2, rocprim::plus<custom_float2>)
    CREATE_BENCHMARK(double2, rocprim::plus<double2>)
    CREATE_BENCHMARK(custom_double2, rocprim::plus<custom_double2>)
    CREATE_BENCHMARK(int8_t, rocprim::plus<int8_t>)
    CREATE_BENCHMARK(uint8_t, rocprim::plus<uint8_t>)
    CREATE_BENCHMARK(rocprim::half, rocprim::plus<rocprim::half>)

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
