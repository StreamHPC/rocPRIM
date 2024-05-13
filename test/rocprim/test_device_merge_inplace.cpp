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
#include "test_utils_assertions.hpp"

#include <rocprim/device/device_merge_inplace.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <gtest/gtest.h>

#include <hip/driver_types.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <limits>

TEST(RocprimDeviceMergeInplaceTests, Basic)
{
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
        };
    }
};

struct large_sizes
{
    std::vector<std::tuple<size_t, size_t>> operator()()
    {
        return {
            std::make_tuple(34567, (1 << 17) - 1220),
            std::make_tuple((1 << 22) - 1652, (1 << 22) - 5839),
            std::make_tuple((1 << 27) - 2459, (1 << 23) - 2134),
            std::make_tuple((1 << 27) - 9532, (1 << 27) - 8421),
        };
    }
};

template<typename T, T start, T increment>
struct linear_data_generator
{
    auto get_iterator()
    {
        return rocprim::make_transform_iterator(rocprim::make_counting_iterator(0),
                                                [](T v) { return v * increment + start; });
    }

    auto get_max_size()
    {
        return increment == 0
                   ? std::numeric_limits<size_t>::max()
                   : static_cast<size_t>((std::numeric_limits<T>::max() - start) / abs(increment));
    }
};

template<typename T, T increment>
struct random_data_generator
{
    struct random_monotonic
    {
        using difference_type = std::ptrdiff_t;
        using value_type      = T;
        using dist_type       = std::conditional_t<std::is_integral<T>::value,
                                             std::uniform_int_distribution<T>,
                                             std::uniform_real_distribution<T>>;

        std::mt19937 engine{std::random_device{}()};
        dist_type    dist{0, increment};

        value_type value = 0;

        int operator*() const
        {
            return value;
        }

        random_monotonic& operator++()
        {
            // prefix
            value += dist(engine);
            return *this;
        }

        random_monotonic operator++(int)
        {
            // postfix
            random_monotonic retval{*this};
            value += dist(engine);
            return retval;
        }
    };

    auto get_iterator()
    {
        return random_monotonic{};
    }

    auto get_max_size()
    {
        return static_cast<size_t>(std::numeric_limits<T>::max() / increment);
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
                             linear_data_generator<int64_t, 0, 2>,
                             linear_data_generator<int64_t, 1, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 2>,
                             linear_data_generator<int32_t, 1, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             linear_data_generator<int16_t, 0, 2>,
                             linear_data_generator<int16_t, 1, 2>>,
    // linear edge cases
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 1>,
                             linear_data_generator<int32_t, 0, 4>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 4>,
                             linear_data_generator<int32_t, 0, 1>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 128, 0>,
                             linear_data_generator<int32_t, 0, 1>>,
    DeviceMergeInplaceParams<int32_t,
                             linear_data_generator<int32_t, 0, 1>,
                             linear_data_generator<int32_t, 128, 0>>,
    // random data
    DeviceMergeInplaceParams<int64_t,
                             random_data_generator<int64_t, 2>,
                             random_data_generator<int64_t, 2>>,
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2>,
                             random_data_generator<int32_t, 2>>,
    DeviceMergeInplaceParams<int16_t,
                             random_data_generator<int16_t, 2>,
                             random_data_generator<int16_t, 2>>,
    DeviceMergeInplaceParams<float,
                             random_data_generator<int32_t, 2>,
                             random_data_generator<int32_t, 2>>,
    // large input sizes
    DeviceMergeInplaceParams<int32_t,
                             random_data_generator<int32_t, 2>,
                             random_data_generator<int32_t, 2>,
                             large_sizes>,
    DeviceMergeInplaceParams<int64_t,
                             random_data_generator<int64_t, 2>,
                             random_data_generator<int64_t, 2>,
                             large_sizes>>
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

    for(auto size : sizes)
    {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        size_t size_a = std::get<0>(size);
        size_t size_b = std::get<1>(size);

        // hipMallocManaged() currently doesnt support zero byte allocation
        if((size_a == 0 || size_b == 0) && test_common_utils::use_hmm())
            continue;

        auto gen_a = gen_a_type{};
        auto gen_b = gen_b_type{};

        // don't test sizes more than we can actually generate
        if(size_a > gen_a.get_max_size() || size_b > gen_b.get_max_size())
            continue;

        auto gen_a_it = gen_a.get_iterator();
        auto gen_b_it = gen_b.get_iterator();

        std::vector<value_type> h_data(size_a + size_b);

        // generate left array
        for(size_t i = 0; i < size_a; ++i)
        {
            h_data[i] = static_cast<value_type>(*(gen_a_it++));
        }

        // generate right array
        for(size_t i = 0; i < size_b; ++i)
        {
            h_data[size_a + i] = static_cast<value_type>(*(gen_b_it++));
        }

        // move input to device
        value_type* d_data;
        size_t      total_bytes = sizeof(value_type) * (size_a + size_b);
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_data, total_bytes));

        HIP_CHECK(
            hipMemcpyWithStream(d_data, h_data.data(), total_bytes, hipMemcpyHostToDevice, stream));

        // allocate temporary storage
        void*  d_temp_storage = nullptr;
        size_t storage_size   = 0;
        HIP_CHECK(rocprim::merge_inplace(d_temp_storage,
                                         storage_size,
                                         d_data,
                                         size_a,
                                         size_b,
                                         compare_op,
                                         stream));
        HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, storage_size));

        // run merge in place
        HIP_CHECK(rocprim::merge_inplace(d_temp_storage,
                                         storage_size,
                                         d_data,
                                         size_a,
                                         size_b,
                                         compare_op,
                                         stream));

        // compare with reference
        std::vector<value_type> h_output(size_a + size_b);
        HIP_CHECK(hipMemcpyWithStream(h_output.data(),
                                      d_data,
                                      total_bytes,
                                      hipMemcpyDeviceToHost,
                                      stream));

        // compute reference
        std::vector<value_type> h_reference(size_a + size_b);
        std::merge(h_data.begin(),
                   h_data.begin() + size_a,
                   h_data.begin() + size_a,
                   h_data.end(),
                   h_reference.begin());

        ASSERT_NO_FATAL_FAILURE((test_utils::assert_eq(h_output, h_reference)));

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));

        HIP_CHECK(hipFree(d_data));
        HIP_CHECK(hipFree(d_temp_storage));
    }
}