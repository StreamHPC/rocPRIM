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

#include "indirect_iterator.hpp"
#include "test_utils_custom_test_types.hpp"
#include "test_utils_types.hpp"

#include <rocprim/detail/various.hpp>
#include <rocprim/device/device_adjacent_find.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/discard_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>
#include <rocprim/types.hpp>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <vector>

// Params for tests
template<class InputType,
         class OutputType         = std::size_t,
         class OpType             = rocprim::equal_to<InputType>,
         class Config             = rocprim::default_config,
         bool UseGraphs           = false,
         bool UseIndirectIterator = false>
struct DeviceAdjacentFindParams
{
    using input_type                            = InputType;
    using output_type                           = OutputType;
    using op_type                               = OpType;
    using config                                = Config;
    static constexpr bool use_graphs            = UseGraphs;
    static constexpr bool use_indirect_iterator = UseIndirectIterator;
};

template<class Params>
class RocprimDeviceAdjacentFindTests : public ::testing::Test
{
public:
    using input_type                            = typename Params::input_type;
    using output_type                           = typename Params::output_type;
    using op_type                               = typename Params::op_type;
    using config                                = typename Params::config;
    static constexpr bool use_graphs            = Params::use_graphs;
    static constexpr bool use_indirect_iterator = Params::use_indirect_iterator;
    static constexpr bool debug_synchronous     = false;
};

// Custom types
using custom_int2        = test_utils::custom_test_type<int>;
using custom_double2     = test_utils::custom_test_type<double>;
using custom_int64_array = test_utils::custom_test_array_type<std::int64_t, 4>;

// Custom configs
using custom_config_0 = rocprim::adjacent_find_config<128, 4>;

using RocprimDeviceAdjacentFindTestsParams = ::testing::Types<
    // Tests with default configuration
    DeviceAdjacentFindParams<int8_t>,
    DeviceAdjacentFindParams<int>,
    DeviceAdjacentFindParams<rocprim::half>,
    DeviceAdjacentFindParams<rocprim::bfloat16>,
    DeviceAdjacentFindParams<float>,
    DeviceAdjacentFindParams<double>,
    // Tests for custom types
    DeviceAdjacentFindParams<custom_int2>,
    DeviceAdjacentFindParams<custom_double2>,
    DeviceAdjacentFindParams<custom_int64_array>,
    // Tests for supported config structs
    DeviceAdjacentFindParams<rocprim::bfloat16,
                             std::size_t,
                             rocprim::equal_to<rocprim::bfloat16>,
                             custom_config_0>,
    // Tests for hipGraph support
    DeviceAdjacentFindParams<unsigned int,
                             std::size_t,
                             rocprim::equal_to<unsigned int>,
                             rocprim::default_config,
                             true>,
    // Tests for when output's value_type is void
    DeviceAdjacentFindParams<int,
                             std::size_t,
                             rocprim::equal_to<int>,
                             rocprim::default_config,
                             false,
                             true>>;

TYPED_TEST_SUITE(RocprimDeviceAdjacentFindTests, RocprimDeviceAdjacentFindTestsParams);

TYPED_TEST(RocprimDeviceAdjacentFindTests, AdjacentFind)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id = " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T                                     = typename TestFixture::input_type;
    using output_type                           = typename TestFixture::output_type;
    using op_type                               = typename TestFixture::op_type;
    static constexpr bool use_indirect_iterator = TestFixture::use_indirect_iterator;
    const bool            debug_synchronous     = TestFixture::debug_synchronous;
    using Config                                = typename TestFixture::config;

    op_type op{};

    for(std::size_t seed_index = 0; seed_index < random_seeds_count + seed_size; seed_index++)
    {
        const unsigned int seed_value
            = seed_index < random_seeds_count ? rand() : seeds[seed_index - random_seeds_count];
        SCOPED_TRACE(testing::Message() << "with seed = " << seed_value);

        for(auto size : test_utils::get_sizes(seed_value))
        {
            hipStream_t stream = 0; // default
            if(TestFixture::use_graphs)
            {
                // Default stream does not support hipGraph stream capture, so create one
                HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            }

            SCOPED_TRACE(testing::Message() << "with size = " << size);

            // Get random index for first adjacent pair
            std::size_t first_adj_index = 0;
            if(size > 1)
            {
                first_adj_index
                    = std::min(test_utils::get_random_value<std::size_t>(
                                   0,
                                   static_cast<size_t>(test_utils::numeric_limits<T>::max()),
                                   seed_value),
                               size - 2);
            }
            SCOPED_TRACE(testing::Message() << "with first_adj_index = " << first_adj_index);

            // Generate input values
            std::vector<T> input(size);
            std::iota(input.begin(), input.begin() + first_adj_index, 0);
            std::fill(input.begin(), input.end(), first_adj_index);

            T*           d_input;
            output_type* d_output;
            HIP_CHECK(
                test_common_utils::hipMallocHelper(&d_input, input.size() * sizeof(*d_input)));
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_output, sizeof(*d_output)));
            HIP_CHECK(hipMemcpy(d_input,
                                input.data(),
                                input.size() * sizeof(*d_input),
                                hipMemcpyHostToDevice));

            const auto output_it
                = test_utils::wrap_in_identity_iterator<use_indirect_iterator>(d_output);

            // Allocate temporary storage
            std::size_t tmp_storage_size;
            void*       d_tmp_storage = nullptr;
            HIP_CHECK(::rocprim::adjacent_find<Config>(d_tmp_storage,
                                                       tmp_storage_size,
                                                       d_input,
                                                       output_it,
                                                       size,
                                                       op,
                                                       stream,
                                                       debug_synchronous));
            ASSERT_GT(tmp_storage_size, 0);
            HIP_CHECK(test_common_utils::hipMallocHelper(&d_tmp_storage, tmp_storage_size));

            hipGraph_t graph;
            if(TestFixture::use_graphs)
            {
                graph = test_utils::createGraphHelper(stream);
            }

            // Run
            HIP_CHECK(::rocprim::adjacent_find<Config>(d_tmp_storage,
                                                       tmp_storage_size,
                                                       d_input,
                                                       output_it,
                                                       size,
                                                       op,
                                                       stream,
                                                       debug_synchronous));

            hipGraphExec_t graph_instance;
            if(TestFixture::use_graphs)
            {
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
            }

            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // Allocate memory for output and copy to host side
            output_type output;
            HIP_CHECK(hipMemcpy(&output, d_output, sizeof(*d_output), hipMemcpyDeviceToHost));

            // Calculate expected results on host
            const auto expected
                = (size > 1) ? std::adjacent_find(input.cbegin(), input.cend(), op) - input.begin()
                             : size;

            // Check if output values are as expected
            ASSERT_EQ(output, expected);

            // Cleanup
            HIP_CHECK(hipFree(d_input));
            HIP_CHECK(hipFree(d_output));
            HIP_CHECK(hipFree(d_tmp_storage));

            if(TestFixture::use_graphs)
            {
                test_utils::cleanupGraphHelper(graph, graph_instance);
                HIP_CHECK(hipStreamDestroy(stream));
            }
        }
    }
}