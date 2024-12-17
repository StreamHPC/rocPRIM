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

// required rocprim headers
#include <rocprim/device/device_scan.hpp>
#include <rocprim/functional.hpp>

// required test headers
#include "test_utils_types.hpp"

#ifndef TEST_FUNC
    #define TEST_FUNC test0
#endif

void TEST_FUNC(size_t size)
{
    using T = int;

    const int seed_value = 123;

    // Generate data
    std::vector<T> input = test_utils::get_random_data<T>(size, 0, 100, seed_value);
    std::vector<T> output(size);

    T* d_input;
    T* d_output;
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(T)));
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, output.size() * sizeof(T)));
    HIP_CHECK(hipMemcpy(d_input, input.data(), input.size() * sizeof(T), hipMemcpyHostToDevice));

    std::vector<T> expected(size);
    // Calculate expected results on host
    test_utils::host_inclusive_scan(input.begin(),
                                    input.end(),
                                    expected.begin(),
                                    rocprim::plus<T>());

    // temp storage
    size_t temp_storage_size_bytes;
    void*  d_temp_storage = nullptr;
    // Get size of d_temp_storage
    HIP_CHECK(rocprim::inclusive_scan(d_temp_storage,
                                      temp_storage_size_bytes,
                                      d_input,
                                      d_output,
                                      input.size(),
                                      rocprim::plus<T>()));

    // temp_storage_size_bytes must be >0
    ASSERT_GT(temp_storage_size_bytes, 0);

    // allocate temporary storage
    HIP_CHECK(test_common_utils::hipMallocHelper(&d_temp_storage, temp_storage_size_bytes));

    // Run
    HIP_CHECK(rocprim::inclusive_scan(d_temp_storage,
                                      temp_storage_size_bytes,
                                      d_input,
                                      d_output,
                                      input.size(),
                                      rocprim::plus<T>()));

    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Copy output to host
    HIP_CHECK(hipMemcpy(output.data(), d_output, output.size() * sizeof(T), hipMemcpyDeviceToHost));

    // Check if output values are as expected
    ASSERT_NO_FATAL_FAILURE(
        test_utils::assert_near(output, expected, test_utils::precision<T> * size));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    HIP_CHECK(hipFree(d_temp_storage));
}
