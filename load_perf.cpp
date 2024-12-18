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

template <typename T>
auto get_random_data_int(size_t size) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    std::uniform_int_distribution<T> distribution;
    std::vector<T> data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <unsigned int BlockSize, unsigned int ItemsPerThread, typename T>
__global__ ROCPRIM_LAUNCH_BOUNDS(BlockSize)
void load(T* input) {
    const unsigned int flat_id = threadIdx.x;
    const unsigned int block_id = blockIdx.x;

    const T* keys_input = input + BlockSize * ItemsPerThread * block_id;
    T keys[ItemsPerThread];

    rocprim::block_load_direct_unordered<BlockSize>(flat_id, keys_input, keys);
    // rocprim::block_load_direct_blocked_vectorized(flat_id, keys_input, keys);
    // rocprim::block_load_direct_striped<BlockSize>(flat_id, keys_input, keys);
    // rocprim::block_load_direct_blocked(flat_id, keys_input, keys);
    // rocprim::block_load_direct_warp_striped(flat_id, keys_input, keys);

    ROCPRIM_UNROLL
    for (unsigned int i = 0; i < ItemsPerThread; ++i) {
        asm volatile ("" :: "r"(keys[i]));
    }
}

template <unsigned int BlockSize, unsigned int ItemsPerThread, typename T>
hipError_t launch_load(T* input, size_t input_size) {
    assert(input_size % (BlockSize * ItemsPerThread) == 0);
    const size_t blocks = input_size / (BlockSize * ItemsPerThread);

    load<BlockSize, ItemsPerThread><<<blocks, BlockSize>>>(input);
    return hipGetLastError();
}

int main() {
    constexpr int block_size = 1024;
    constexpr int items_per_thread = 16;

    using Key = uint16_t;
    constexpr size_t size = 268'435'456;

    Key* d_input;
    Key* d_output;
    HIP_CHECK(hipMalloc(&d_input, size * sizeof(Key)));
    HIP_CHECK(hipMalloc(&d_output, size * sizeof(Key)));

    std::cout << "generating data " << size << std::endl;

    const auto input = get_random_data_int<Key>(size);

    std::cout << "allocating input" << std::endl;

    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(Key), hipMemcpyHostToDevice));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    std::cout << "\nwarming up" << std::endl;

    for (int i = 0; i < 5; ++i) {
        HIP_CHECK((launch_load<block_size, items_per_thread>(d_input, size)));
    }

    std::cout << "benchmarking" << std::endl;

    HIP_CHECK(hipEventRecord(start, hipStreamDefault));

    const size_t rounds = 100;

    for (int i = 0; i < rounds; ++i) {
        HIP_CHECK((launch_load<block_size, items_per_thread>(d_input, size)));
    }

    HIP_CHECK(hipEventRecord(stop, hipStreamDefault));
    HIP_CHECK(hipEventSynchronize(stop));

    std::cout << std::endl;

    float elapsed_ms;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= rounds;
    float elapsed_s = elapsed_ms / 1'000;
    std::cout << "time per round: " << std::fixed << std::setprecision(5) << elapsed_ms * 1000'000 << " us" << std::endl;

    const float movement_current =
        size * sizeof(Key); // Loads, histogram
    std::cout << std::fixed << std::setprecision(5) << "current movement: " << movement_current / elapsed_s / 1000'000'000.f << "GB/s" << std::endl;
}
