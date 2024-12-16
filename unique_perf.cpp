#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>

#define HIP_CHECK(x) assert((x) == hipSuccess);

auto get_random_data(size_t size, float min, float max) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<float> distribution(min, max);
    std::vector<rocprim::half> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<rocprim::half>(distribution(gen)); });
    return data;
}

int main() {
    using Key = rocprim::half;
    constexpr size_t size = 1024 * 1024;

    Key* d_input;
    Key* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(Key)));

    const auto input = get_random_data(size, -1000, 1000);

    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(Key), hipMemcpyHostToDevice));

    void* d_temporary_storage = nullptr;
    size_t temp_storage_size = 0;

    HIP_CHECK(rocprim::radix_sort_keys(
        d_temporary_storage,
        temp_storage_size,
        d_input,
        d_output,
        size
    ));

    HIP_CHECK(hipMalloc(&d_temporary_storage, temp_storage_size));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < 50; ++i) {
        HIP_CHECK(rocprim::radix_sort_keys(
            d_temporary_storage,
            temp_storage_size,
            d_input,
            d_output,
            size
        ));
    }

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    for (int i = 0; i < 100; ++i) {
        HIP_CHECK(rocprim::radix_sort_keys(
            d_temporary_storage,
            temp_storage_size,
            d_input,
            d_output,
            size
        ));
    }

    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_mseconds;
    HIP_CHECK(hipEventElapsedTime(&elapsed_mseconds, start, stop));
    std::cout << std::setprecision(100) << elapsed_mseconds * 1000'000 / 100 << std::endl;
}
