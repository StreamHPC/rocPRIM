#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#define HIP_CHECK(x) assert((x) == hipSuccess);

auto get_random_data_f16(size_t size, float min, float max) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);
    std::vector<rocprim::half> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<rocprim::half>(distribution(gen)); });
    return data;
}

auto get_random_data_u8(size_t size) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<uint8_t> distribution;
    std::vector<uint8_t> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<uint8_t>(distribution(gen)); });
    return data;
}

int main() {
    using Key = rocprim::half;
    constexpr size_t size = 1024 * 1024 * 32;

    Key* d_input;
    Key* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(Key)));

    std::cout << "generating data" << std::endl;

    const auto input = get_random_data_f16(size, -1000, 1000);

    std::cout << "allocating input" << std::endl;

    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(Key), hipMemcpyHostToDevice));

    void* d_temporary_storage = nullptr;
    size_t temp_storage_size = 0;

    unsigned bits = 8 * sizeof(Key);

    HIP_CHECK(rocprim::radix_sort_keys(
        d_temporary_storage,
        temp_storage_size,
        d_input,
        d_output,
        size,
        0,
        bits
    ));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temp_storage_size));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::cout << "debug_synchronous output:\n" << std::endl;

    HIP_CHECK(rocprim::radix_sort_keys(
        d_temporary_storage,
        temp_storage_size,
        d_input,
        d_output,
        size,
        0,
        bits,
        hipStreamDefault,
        true
    ));

    std::cout << "\nwarming up" << std::endl;

    for (int i = 0; i < 50; ++i) {
        HIP_CHECK(rocprim::radix_sort_keys(
            d_temporary_storage,
            temp_storage_size,
            d_input,
            d_output,
            size,
            0,
            bits
        ));
    }

    std::cout << "benchmarking" << std::endl;

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    const size_t rounds = 100;

    for (int i = 0; i < rounds; ++i) {
        HIP_CHECK(rocprim::radix_sort_keys(
            d_temporary_storage,
            temp_storage_size,
            d_input,
            d_output,
            size,
            0,
            bits
        ));
    }

    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    std::cout << "hot debug_synchronous output:\n" << std::endl;

    HIP_CHECK(rocprim::radix_sort_keys(
        d_temporary_storage,
        temp_storage_size,
        d_input,
        d_output,
        size,
        0,
        bits,
        hipStreamDefault,
        true
    ));

    std::cout << std::endl;

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= rounds;
    float elapsed_s = elapsed_ms / 1'000;
    std::cout << "time per round: " << std::fixed << std::setprecision(5) << elapsed_ms * 1000'000 << " us" << std::endl;

    const float movement_apparent =
        size * sizeof(Key) // Loads
        + size * sizeof(Key); // stores;

    const float movement_total =
        size * sizeof(Key) // Loads, histogram
        + (
            size * sizeof(Key) // Loads, iteration
            + size * sizeof(Key) // Stores, iteration
        ) * sizeof(Key); // Groups of 8 bits

    const float movement_current =
        size * sizeof(Key); // Loads, histogram

    std::cout << std::fixed << std::setprecision(5) << "apparent movement: " << movement_apparent / elapsed_s / 1000'000'000.f << "GB/s" << std::endl;
    std::cout << std::fixed << std::setprecision(5) << "total movement: " << movement_total / elapsed_s / 1000'000'000.f << "GB/s" << std::endl;
    std::cout << std::fixed << std::setprecision(5) << "current movement: " << movement_current / elapsed_s / 1000'000'000.f << "GB/s" << std::endl;
}
