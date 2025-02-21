// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_

#include <type_traits>
#include <iterator>

#include "../../config.hpp"
#include "../../detail/various.hpp"

#include "../../intrinsics.hpp"
#include "../../functional.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_store.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

// Wrapper for unpacking tuple to be used with BinaryFunction.
// See transform function which accepts two input iterators.
template<class T1, class T2, class BinaryFunction>
struct unpack_binary_op
{
    using result_type = typename ::rocprim::invoke_result<BinaryFunction, T1, T2>::type;

    ROCPRIM_HOST_DEVICE inline
    unpack_binary_op() = default;

    ROCPRIM_HOST_DEVICE inline
    unpack_binary_op(BinaryFunction binary_op) : binary_op_(binary_op)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~unpack_binary_op() = default;

    ROCPRIM_HOST_DEVICE inline
    result_type operator()(const ::rocprim::tuple<T1, T2>& t)
    {
        return binary_op_(::rocprim::get<0>(t), ::rocprim::get<1>(t));
    }

private:
    BinaryFunction binary_op_;
};

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class ResultType,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto transform_kernel_impl(InputIterator  input,
                           const size_t   input_size,
                           OutputIterator output,
                           UnaryFunction  transform_op) ->
    typename std::enable_if_t<std::is_pointer<InputIterator>::value
                              && std::is_pointer<OutputIterator>::value>
// && (sizeof(typename std::iterator_traits<InputIterator>::value_type) >= 4)
// && (sizeof(typename std::iterator_traits<InputIterator>::value_type) % 4 == 0)>
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, ResultType, output_type
        >::type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    constexpr unsigned int vectors_per_thread
        = sizeof(input_type) * ItemsPerThread / sizeof(uint128_t);

    // static_assert(vectors_per_thread > 0, "The input type should have at least number of items that fit in 1 uint128_t");
    // if (vectors_per_thread == 0) printf("%d %d %d\n", sizeof(input_type), ItemsPerThread, BlockSize);
    // assert(vectors_per_thread > 0);

    ROCPRIM_SHARED_MEMORY
    uint128_t shared_values[BlockSize * vectors_per_thread];

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = ::rocprim::detail::grid_size<0>();
    const unsigned int valid_in_last_block = input_size - block_offset;

    input_type input_values[ItemsPerThread];
    result_type output_values[ItemsPerThread];

    // TODO: Cut off first and last parts, set as blocks 0 and 1, then proceed

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            input_values,
            valid_in_last_block
        );

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_in_last_block)
            {
                output_values[i] = transform_op(input_values[i]);
            }
        }

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            output_values,
            valid_in_last_block
        );
    }
    else
    {
        input_type*  block_input  = input + block_offset;
        output_type* block_output = output + block_offset;

        constexpr unsigned int warp_size = rocprim::device_warp_size();

        unsigned int lane_id     = detail::logical_lane_id<warp_size>();
        unsigned int warp_id     = flat_id / warp_size;
        unsigned int warp_offset = warp_id * warp_size * vectors_per_thread;

        const uint128_t* vector_ptr
            = reinterpret_cast<const uint128_t*>(block_input) + warp_offset + lane_id;

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < vectors_per_thread; item++)
        {
            shared_values[item + flat_id * vectors_per_thread] = thread_load<load_nontemporal>(
                vector_ptr + (item * warp_size)); // Change this to direct load to LDS
        }

        const input_type* values
            = reinterpret_cast<const input_type*>(&shared_values[flat_id * vectors_per_thread]);

        syncthreads(); // Probably less sync needed

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < ItemsPerThread; item++)
        {
            output_values[item] = transform_op(values[item]);
        }

        constexpr unsigned int vectors_per_thread_output
            = rocprim::max((sizeof(output_type) * ItemsPerThread) / sizeof(uint128_t), 1ul);

        unsigned int warp_offset_output = warp_id * warp_size * vectors_per_thread_output;

        rocprim::uint128_t* vector_ptr_output
            = reinterpret_cast<rocprim::uint128_t*>(block_output) + warp_offset + lane_id;

        ROCPRIM_UNROLL
        for(unsigned int item = 0; item < vectors_per_thread_output; item++)
        {
            vector_ptr_output[item * warp_size]
                = *(reinterpret_cast<const rocprim::uint128_t*>(output_values) + item);
        }
    }
}

template<unsigned int BlockSize,
         unsigned int ItemsPerThread,
         class ResultType,
         class InputIterator,
         class OutputIterator,
         class UnaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE
auto transform_kernel_impl(InputIterator  input,
                           const size_t   input_size,
                           OutputIterator output,
                           UnaryFunction  transform_op) ->
    typename std::enable_if_t<
        !std::is_pointer<InputIterator>::value
        || !std::is_pointer<OutputIterator>::value>
// || !(sizeof(typename std::iterator_traits<InputIterator>::value_type) >= 4)
// || !(sizeof(typename std::iterator_traits<InputIterator>::value_type) % 4 == 0)>
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using output_type = typename std::iterator_traits<OutputIterator>::value_type;
    using result_type =
        typename std::conditional<
            std::is_void<output_type>::value, ResultType, output_type
        >::type;

    constexpr unsigned int items_per_block = BlockSize * ItemsPerThread;

    const unsigned int flat_id = ::rocprim::detail::block_thread_id<0>();
    const unsigned int flat_block_id = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset = flat_block_id * items_per_block;
    const unsigned int number_of_blocks = ::rocprim::detail::grid_size<0>();
    const unsigned int valid_in_last_block = input_size - block_offset;

    input_type input_values[ItemsPerThread];
    result_type output_values[ItemsPerThread];

    if(flat_block_id == (number_of_blocks - 1)) // last block
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            input_values,
            valid_in_last_block
        );

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            if(BlockSize * i + flat_id < valid_in_last_block)
            {
                output_values[i] = transform_op(input_values[i]);
            }
        }

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            output_values,
            valid_in_last_block
        );
    }
    else
    {
        block_load_direct_striped<BlockSize>(
            flat_id,
            input + block_offset,
            input_values
        );

        ROCPRIM_UNROLL
        for(unsigned int i = 0; i < ItemsPerThread; i++)
        {
            output_values[i] = transform_op(input_values[i]);
        }

        block_store_direct_striped<BlockSize>(
            flat_id,
            output + block_offset,
            output_values
        );
    }
}

} // end of detail namespace

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_TRANSFORM_HPP_
