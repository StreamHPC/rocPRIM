// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../common_test_header.hpp"
#include "test_seed.hpp"
#include "test_utils_assertions.hpp"
#include "test_utils_data_generation.hpp"

#include <hip/amd_detail/amd_hip_runtime.h>
#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_merge_inplace.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <gtest/gtest.h>

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>

TEST(RocprimDeviceMergeInplaceTests, Basic)
{
    bool has_cooperative_launching = false;
    rocprim::detail::supports_cooperative_groups(has_cooperative_launching, 0);
    if(!has_cooperative_launching)
    {
        GTEST_SKIP() << "This device does not support cooperative groups.";
    }

    using value_t            = int;
    void*  temporary_storage = nullptr;
    size_t storage_size      = 0;

    int left_size  = 128 * 1024;
    int right_size = 128 * 1024;

    int total_size = left_size + right_size;

    std::vector<value_t> h_data(total_size);

    for(int i = 0; i < left_size; ++i)
    {
        h_data[i] = i * 1;
    }

    for(int i = 0; i < right_size; ++i)
    {
        h_data[left_size + i] = i * 2 + 1;
    }

    std::vector<value_t> h_expected(h_data);

    size_t   size_bytes = sizeof(value_t) * (left_size + right_size);
    value_t* d_data;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_data, size_bytes));

    HIP_CHECK(hipMemcpy(d_data, h_data.data(), size_bytes, hipMemcpyHostToDevice));

    // get temporary storage
    HIP_CHECK(
        rocprim::merge_inplace(temporary_storage, storage_size, d_data, left_size, right_size));

    HIP_CHECK(test_common_utils::hipMallocHelper(&temporary_storage, storage_size));

    HIP_CHECK(
        rocprim::merge_inplace(temporary_storage, storage_size, d_data, left_size, right_size));

    std::inplace_merge(h_expected.begin(), h_expected.begin() + left_size, h_expected.end());

    HIP_CHECK(hipMemcpy(h_data.data(), d_data, size_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(temporary_storage));
    HIP_CHECK(hipFree(d_data));

    test_utils::assert_eq(h_expected, h_data);
}

struct small_sizes
{
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple(0, 0),
            std::make_tuple(2, 1),
            std::make_tuple(10, 10),
            std::make_tuple(111, 111),
            std::make_tuple(128, 1289),
            std::make_tuple(12, 1000),
            std::make_tuple(123, 3000),
            std::make_tuple(1024, 512),
            std::make_tuple(2345, 49),
            std::make_tuple(17867, 41),
            std::make_tuple(17867, 34567),
            std::make_tuple(924353, 1723454),
            std::make_tuple(123, 33554432),
            std::make_tuple(33554432, 123),
            std::make_tuple(33554432, 33554432),
            std::make_tuple(34567, (1 << 17) - 1220),
        };
    }
};

struct large_sizes
{
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple((1 << 14) - 1652, (1 << 27) - 5839),
            std::make_tuple((1 << 27) - 2459, (1 << 14) - 2134),
            std::make_tuple((1 << 28) - 9532, (1 << 28) - 8421),
        };
    }
};

struct extreme_sizes
{
    // generating the data and checking the data on this order on the CPU is slow.
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple((1ULL << 32) + 5327, (1ULL << 32) + 9682),
        };
    }
};

template<typename T, T start, T increment>
struct linear_data_generator
{
    static constexpr bool is_random = false;

    auto get_iterator(seed_type /* seed */)
    {
        return rocprim::make_transform_iterator(rocprim::make_counting_iterator(T(0)),
                                                [](T v) { return v * increment + start; });
    }

    auto get_max_size()
    {
        return increment == 0
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>((std::numeric_limits<T>::max() - start) / abs(increment));
    }
};

template<typename T, T increment, int max_duplicates>
struct random_data_generator
{
    static constexpr bool is_random = true;

    struct random_monotonic
    {
        using difference_type = std::ptrdiff_t;
        using value_type      = T;

        // not all integral types are valid for int distribution
        using dist_value_type = std::conditional_t<
            std::is_integral<T>::value
                && !test_utils::is_valid_for_int_distribution<value_type>::value,
            int,
            value_type>;

        using val_dist_type = std::conditional_t<std::is_integral<T>::value,
                                                 std::uniform_int_distribution<dist_value_type>,
                                                 std::uniform_real_distribution<dist_value_type>>;
        using dup_dist_type = std::uniform_int_distribution<int>;

        seed_type seed;
        int       duplicates;

        std::minstd_rand engine{seed};
        val_dist_type val_dist{dist_value_type{1}, dist_value_type{increment}};
        dup_dist_type dup_dist{dist_value_type{1}, dist_value_type{max_duplicates}};

        dist_value_type value = dist_value_type{0};

        random_monotonic(seed_type seed) : seed(seed) {}

        int operator*() const
        {
            return test_utils::saturate_cast<value_type>(value);
        }

        void next()
        {
            // consume a duplicate
            duplicates--;

            // if we have duplicates left over, do nothing
            if(duplicates > 0 || value >= std::numeric_limits<value_type>::max() - increment)
                return;

            // get new duplicates
            duplicates = max_duplicates > 1 ? dup_dist(engine) : 1;
            value += val_dist(engine);
        }

        random_monotonic& operator++()
        {
            // prefix
            next();
            return *this;
        }

        random_monotonic operator++(int)
        {
            // postfix
            random_monotonic retval{*this};
            next();
            return retval;
        }
    };

    auto get_iterator(seed_type seed)
    {
        return random_monotonic{seed};
    }

    auto get_max_size()
    {
        return static_cast<size_t>(std::numeric_limits<size_t>::max());
    }
};

template<typename ValueType,
         typename GenAType,
         typename GenBType,
         typename Sizes     = small_sizes,
         typename CompareOp = ::rocprim::less<ValueType>,
         bool UseGraphs     = false>
struct DeviceMergeInplaceParams
{
    using value_type = ValueType;
    using gen_a_type = GenAType;
    using gen_b_type = GenBType;
    using sizes      = Sizes;
};

typedef ::testing::Types<
    // linear even-odd
    DeviceMergeInplaceParams<int64_t,
                             linear_data_generator<int64_t, 1, 2>,
                             linear_data_generator<int64_t, 0, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 1, 2>,
                             linear_data_generator<int32_t, 0, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             linear_data_generator<int16_t, 1, 2>,
                             linear_data_generator<int16_t, 0, 2>>,
    // linear edge cases
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 1>,
                             linear_data_generator<int32_t, 0, 4>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 4>,
                             linear_data_generator<int32_t, 0, 1>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 5000, 1>,
                             linear_data_generator<int32_t, 0, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 2>,
                             linear_data_generator<int32_t, 5000, 1>>,
    // random data
    DeviceMergeInplaceParams<int64_t,
                             random_data_generator<int64_t, 2, 2>,
                             random_data_generator<int64_t, 2, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2, 2>,
                             random_data_generator<int32_t, 2, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             random_data_generator<int16_t, 2, 2>,
                             random_data_generator<int16_t, 2, 2>>,
    DeviceMergeInplaceParams<float,
                             random_data_generator<int32_t, 2, 2>,
                             random_data_generator<int32_t, 2, 2>>,
    // large input sizes
    DeviceMergeInplaceParams<int8_t,
                             random_data_generator<int8_t, 1, 1 << 27>,
                             random_data_generator<int8_t, 1, 1 << 27>,
                             large_sizes>,
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2, 4>,
                             random_data_generator<int32_t, 2, 4>,
                             large_sizes>
// extreme sizes - disabled because it takes too long
#if TEST_EXTREME_SIZES
    ,
    DeviceMergeInplaceParams<int8_t,
                             random_data_generator<int8_t, 1, 1 << 27>,
                             random_data_generator<int8_t, 1, 1 << 27>,
                             extreme_sizes>
#endif
    >
    DeviceMergeInplaceTestsParams;

template<typename Params>
struct DeviceMergeInplaceTests : public testing::Test
{
    using value_type = typename Params::value_type;
    using gen_a_type = typename Params::gen_a_type;
    using gen_b_type = typename Params::gen_b_type;
    using sizes      = typename Params::sizes;
};

TYPED_TEST_SUITE(DeviceMergeInplaceTests, DeviceMergeInplaceTestsParams);

TYPED_TEST(DeviceMergeInplaceTests, MergeInplace)
{
    using value_type = typename TestFixture::value_type;
    using gen_a_type = typename TestFixture::gen_a_type;
    using gen_b_type = typename TestFixture::gen_b_type;
    using binary_op  = rocprim::less<value_type>;

    auto sizes = typename TestFixture::sizes{}();

    binary_op compare_op{};

    hipStream_t stream = hipStreamDefault;

    bool has_cooperative_launching = false;
    rocprim::detail::supports_cooperative_groups(has_cooperative_launching, stream);
    if(!has_cooperative_launching)
    {
        GTEST_SKIP() << "This device does not support cooperative groups.";
    }

    for(auto size : sizes)
    {
        const size_t num_seeds
            = gen_a_type::is_random || gen_b_type::is_random ? (random_seeds_count + seed_size) : 1;

        const size_t size_a      = std::get<0>(size);
        const size_t size_b      = std::get<1>(size);
        const size_t total_items = size_a + size_b;

        // hipMallocManaged() currently doesnt support zero byte allocation
        if((size_a == 0 || size_b == 0) && test_common_utils::use_hmm())
            continue;

        auto gen_a = gen_a_type{};
        auto gen_b = gen_b_type{};

        // don't test sizes more than we can actually generate
        if(size_a > gen_a.get_max_size() || size_b > gen_b.get_max_size())
            continue;

        const size_t total_bytes  = sizeof(value_type) * total_items;
        size_t       storage_size = 0;

        value_type* d_data         = nullptr;
        void*       d_temp_storage = nullptr;

        HIP_CHECK(rocprim::merge_inplace(d_temp_storage,
                                         storage_size,
                                         d_data,
                                         size_a,
                                         size_b,
                                         compare_op,
                                         stream));

        hipError_t malloc_result = test_common_utils::hipMallocHelper(&d_data, total_bytes);
        if (malloc_result == hipErrorOutOfMemory){
            continue;
        }
        HIP_CHECK(malloc_result);

        malloc_result = test_common_utils::hipMallocHelper(&d_temp_storage, storage_size);
        if (malloc_result == hipErrorOutOfMemory) {
            hipFree(d_data);
            continue;
        }
        HIP_CHECK(malloc_result);

        std::vector<value_type> h_data(size_a + size_b);

        HIP_CHECK(hipDeviceSynchronize());

        for(size_t seed_index = 0; seed_index < num_seeds; seed_index++)
        {
            unsigned int seed_value
                = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
            SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

            auto gen_a_it = gen_a.get_iterator(seed_value);
            auto gen_b_it = gen_b.get_iterator(seed_value + 1);

            // generate left array
            for(size_t i = 0; i < size_a; ++i)
            {
                h_data[i] = static_cast<value_type>(*gen_a_it);
                gen_a_it++;
            }

            // generate right array
            for(size_t i = 0; i < size_b; ++i)
            {
                h_data[size_a + i] = static_cast<value_type>(*gen_b_it);
                gen_b_it++;
            }

            // move input to device
            HIP_CHECK(hipMemcpyWithStream(d_data,
                                          h_data.data(),
                                          total_bytes,
                                          hipMemcpyHostToDevice,
                                          stream));
            // run merge in place
            HIP_CHECK(rocprim::merge_inplace(d_temp_storage,
                                             storage_size,
                                             d_data,
                                             size_a,
                                             size_b,
                                             compare_op,
                                             stream));

            // compare with reference
            std::vector<value_type> h_output(total_items);
            HIP_CHECK(hipMemcpyWithStream(h_output.data(),
                                          d_data,
                                          total_bytes,
                                          hipMemcpyDeviceToHost,
                                          stream));

            // For large inputs, we test for monoticity instead of equality with reference.
            if(total_items > 1 << 20)
            {
                // we keep the previous value so we don't invoke multiple loads
                value_type prev = h_output[0];
                for(size_t i = 1; i < total_items; ++i)
                {
                    value_type current = h_output[i];
                    if(prev > current)
                    {
                        FAIL() << "Result is not monotonic at " << i << ": " << prev
                               << " <= " << current;
                    }

                    prev = current;
                }
            }
            else
            {
                // compute reference
                std::vector<value_type> h_reference(total_items);
                std::merge(h_data.begin(),
                           h_data.begin() + size_a,
                           h_data.begin() + size_a,
                           h_data.end(),
                           h_reference.begin());

                ASSERT_NO_FATAL_FAILURE((test_utils::assert_eq(h_output, h_reference)));
            }
        }
        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}
