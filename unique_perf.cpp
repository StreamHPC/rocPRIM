#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <rocprim/rocprim.hpp>
#include <vector>

#define HIP_CHECK(x) assert((x) == hipSuccess);

using Key = rocprim::half;

auto get_random_data_f16(size_t size, float min, float max)
{
    std::random_device                    rd;
    std::default_random_engine            gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);
    std::vector<rocprim::half>            data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return static_cast<rocprim::half>(distribution(gen)); });
    return data;
}

auto get_random_data_u8(size_t size)
{
    std::random_device                     rd;
    std::default_random_engine             gen(rd());
    std::uniform_int_distribution<uint8_t> distribution;
    std::vector<uint8_t>                   data(size);
    std::generate(data.begin(),
                  data.end(),
                  [&]() { return static_cast<uint8_t>(distribution(gen)); });
    return data;
}

float benchmark(const Key* input, size_t size, bool verbose = false)
{
    std::cout << "====================" << std::endl;
    std::cout << "size: " << size << std::endl;

    Key*          d_input;
    Key*          d_output;
    Key*          d_unique_output;
    unsigned int* d_counts_output;
    size_t*       d_output_count;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_unique_output, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_counts_output, size * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&d_output_count, sizeof(size_t)));

    // std::cout << "generating data" << std::endl;
    // std::cout << "allocating input" << std::endl;

    HIP_CHECK(hipMemcpy(d_input, input, size * sizeof(Key), hipMemcpyHostToDevice));

    void*  d_temporary_storage      = nullptr;
    size_t temp_storage_size_sort   = 0;
    size_t temp_storage_size_unique = 0;

    unsigned bits = 8 * sizeof(Key);

    HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage,
                                       temp_storage_size_sort,
                                       d_input,
                                       d_output,
                                       size,
                                       0,
                                       bits));

    HIP_CHECK(rocprim::run_length_encode(d_temporary_storage,
                                         temp_storage_size_unique,
                                         d_output,
                                         size,
                                         d_unique_output,
                                         d_counts_output,
                                         d_output_count));

    size_t temp_storage_size = rocprim::max(temp_storage_size_sort, temp_storage_size_unique);

    HIP_CHECK(hipMalloc(&d_temporary_storage, temp_storage_size));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage,
                                       temp_storage_size,
                                       d_input,
                                       d_output,
                                       size,
                                       0,
                                       bits,
                                       hipStreamDefault,
                                       false));

    HIP_CHECK(rocprim::unique(d_temporary_storage,
                              temp_storage_size,
                              d_output,
                              d_unique_output,
                              d_output_count,
                              size,
                              rocprim::equal_to<Key>{},
                              hipStreamDefault,
                              false));

    // std::cout << "\nwarming up" << std::endl;

    for(int i = 0; i < 50; ++i)
    {
        HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage,
                                           temp_storage_size,
                                           d_input,
                                           d_output,
                                           size,
                                           0,
                                           bits));

        HIP_CHECK(rocprim::run_length_encode(d_temporary_storage,
                                             temp_storage_size,
                                             d_output,
                                             size,
                                             d_unique_output,
                                             d_counts_output,
                                             d_output_count));
    }

    // std::cout << "benchmarking" << std::endl;

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    const size_t rounds = 100;

    for(int i = 0; i < rounds; ++i)
    {
        HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage,
                                           temp_storage_size,
                                           d_input,
                                           d_output,
                                           size,
                                           0,
                                           bits));
        HIP_CHECK(rocprim::run_length_encode(d_temporary_storage,
                                             temp_storage_size,
                                             d_output,
                                             size,
                                             d_unique_output,
                                             d_counts_output,
                                             d_output_count));
    }

    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    if(verbose)
        std::cout << "hot debug_synchronous output (radix sort):\n" << std::endl;
    HIP_CHECK(rocprim::radix_sort_keys(d_temporary_storage,
                                       temp_storage_size,
                                       d_input,
                                       d_output,
                                       size,
                                       0,
                                       bits,
                                       hipStreamDefault,
                                       verbose));

    if(verbose)
        std::cout << "hot debug_synchronous output (run length encode):\n" << std::endl;
    HIP_CHECK(rocprim::run_length_encode(d_temporary_storage,
                                         temp_storage_size,
                                         d_output,
                                         size,
                                         d_unique_output,
                                         d_counts_output,
                                         d_output_count,
                                         hipStreamDefault,
                                         verbose));

    if(verbose)
        std::cout << std::endl;

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= rounds;
    float elapsed_s = elapsed_ms / 1'000;
    std::cout << "time per round: " << std::fixed << std::setprecision(5) << elapsed_ms * 1000'000
              << " ns" << std::endl;

    const float movement_apparent = size * sizeof(Key) // Loads
                                    + size * sizeof(Key); // stores;

    const float movement_total = size * sizeof(Key) // Loads, histogram
                                 + (size * sizeof(Key) // Loads, iteration
                                    + size * sizeof(Key) // Stores, iteration
                                    ) * sizeof(Key) // Groups of 8 bits
                                 + (size * sizeof(Key)) * 2; // Loads + stores, reduce by key

    const float movement_current = size * sizeof(Key);

    std::cout << std::fixed << std::setprecision(5)
              << "apparent movement: " << movement_apparent / elapsed_s / 1000'000'000.f << "GB/s"
              << std::endl;
    std::cout << std::fixed << std::setprecision(5)
              << "current movement: " << movement_current / elapsed_s / 1000'000'000.f << "GB/s"
              << std::endl;
    std::cout << std::fixed << std::setprecision(5)
              << "total movement: " << movement_total / elapsed_s / 1000'000'000.f << "GB/s"
              << std::endl;

    return elapsed_ms * 1000'0000;
}

int main()
{
    std::cout << "generating input" << std::endl;
    const auto input = get_random_data_f16(1024 * 524288, 0, 1024);

    std::vector<float> times;
    for(int i = 1024; i <= 524288; i *= 2)
    {
        times.push_back(benchmark(input.data(), 1024 * i));
    }

    std::cout << "sizes: ";
    for(int i = 1024; i <= 524288; i *= 2)
    {
        std::cout << "\t" << 1024 * i;
    }
    std::cout << std::endl;

    std::cout << "times: ";
    for(auto t : times)
    {
        std::cout << "\t" << t;
    }
    std::cout << std::endl;
}
