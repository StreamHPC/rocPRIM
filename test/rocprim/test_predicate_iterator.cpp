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

#include "test_utils_data_generation.hpp"

#include <common_test_header.hpp>

#include <rocprim/device/device_transform.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/predicate_iterator.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

template<class T>
struct is_odd
{
    __device__ __host__ constexpr T operator()(const T& a) const
    {
        return a % 2;
    }
};

template<int V, class T>
struct set_to
{
    __device__ __host__ constexpr T operator()(const T&) const
    {
        return V;
    }
};

template<int V, class T>
struct increment_by
{
    __device__ __host__ constexpr T operator()(const T& a) const
    {
        return a + V;
    }
};

TEST(RocprimPredicateIteratorTests, TypeTraits)
{
    using value_type = int;

    value_type* data{};
    bool*       mask{};

    auto it = rocprim::make_mask_iterator(data, mask);

    using it_t      = decltype(it);
    using discard_t = it_t::discard_reference;

    static_assert(std::is_assignable<discard_t, value_type>::value,
                  "discard type is not assignable with underlying type, even though it should be!");
    static_assert(std::is_assignable<decltype(*it), value_type>::value,
                  "iterator is not assignable with underlying type via dereference, even though it "
                  "should be!");
    static_assert(std::is_assignable<decltype(it[0]), value_type>::value,
                  "iterator is not assignablle with underlying type via array index, even though "
                  "is should be!");
}

// Test that we are only writing if predicate holds
TEST(RocprimPredicateIteratorTests, HostWrite)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), 0);

    // Make iterator that only writes to odd values
    auto odd_it = rocprim::make_predicate_iterator(data.begin(), is_odd<T>{});

    // Increment all values in that iterator
    std::transform(data.begin(), data.end(), odd_it, [](auto v) { return v + 1; });

    // Such that none of data is odd
    ASSERT_TRUE(std::none_of(data.begin(), data.end(), is_odd<T>{}));
}

// Test that we are only reading if predicate holds, excluding the required read for the predicate
TEST(RocprimPredicateIteratorTests, HostRead)
{
    using T                      = int;
    static constexpr size_t size = 100;

    auto is_odd_or_default = [](T v) { return v % 2 || v == T{}; };

    std::vector<T> data(size);
    std::iota(data.begin(), data.end(), 0);

    // Make iterator that only reads odd values
    auto odd_it = rocprim::make_predicate_iterator(data.begin(), is_odd<T>{});

    // Read all values from that iterator
    for(size_t i = 0; i < size; ++i)
    {
        data[i] = odd_it[i];
    }

    // Such that all of data is odd or default
    ASSERT_TRUE(std::all_of(data.begin(), data.end(), is_odd_or_default));
}

// Test that we are only writing if predicate holds
TEST(RocprimPredicateIteratorTests, HostMaskWrite)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T>    data(size);
    std::vector<bool> mask = test_utils::get_random_data<bool>(size, false, true, 0);
    std::iota(data.begin(), data.end(), 0);
    test_utils::get_random_data<bool>(size, false, true, 0);

    auto masked_it = rocprim::make_predicate_iterator(data.begin(), mask.begin(), is_odd<T>{});
    std::transform(data.begin(), data.end(), masked_it, set_to<-1, T>{});

    for(size_t i = 0; i < size; ++i)
    {
        if(mask[i])
        {
            ASSERT_EQ(data[i], -1);
        }
        else
        {
            ASSERT_EQ(data[i], i);
        }
    }
}

// Test that we are only reading if predicate holds, excluding the required read for the predicate
TEST(RocprimPredicateIteratorTests, HostMaskRead)
{
    using T                      = int;
    static constexpr size_t size = 100;

    std::vector<T>    data(size);
    std::vector<bool> mask = test_utils::get_random_data<bool>(size, false, true, 0);
    std::iota(data.begin(), data.end(), 0);

    auto masked_it = rocprim::make_mask_iterator(data.begin(), mask.begin());

    for(size_t i = 0; i < size; ++i)
    {
        data[i] = masked_it[i];
    }

    for(size_t i = 0; i < size; ++i)
    {
        if(mask[i])
        {
            ASSERT_EQ(data[i], i);
        }
        else
        {
            ASSERT_EQ(data[i], T{});
        }
    }
}

// Test if predicate iterator can be used on device
TEST(RocprimPredicateIteratorTests, DeviceTransformIf)
{
    using T         = int;
    using predicate = is_odd<T>;
    using transform = increment_by<5, T>;

    constexpr size_t size      = 100;
    constexpr size_t data_size = sizeof(T) * size;

    std::vector<T> h_data(size);
    std::iota(h_data.begin(), h_data.end(), 0);

    T* d_data;
    HIP_CHECK(hipMalloc(&d_data, data_size));
    HIP_CHECK(hipMemcpy(d_data, h_data.data(), data_size, hipMemcpyHostToDevice));

    auto c_it = rocprim::make_counting_iterator(0);
    auto p_it = rocprim::make_predicate_iterator(d_data, c_it, predicate{});

    HIP_CHECK(rocprim::transform(d_data, p_it, size, transform{}));

    HIP_CHECK(hipMemcpy(h_data.data(), d_data, data_size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_data));

    for(size_t i = 0; i < size; ++i)
    {
        if(predicate{}(i))
        {
            ASSERT_EQ(h_data[i], transform{}(i));
        }
        else
        {
            ASSERT_EQ(h_data[i], i);
        }
    }
}