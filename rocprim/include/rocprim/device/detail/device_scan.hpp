// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_

#include <iterator>
#include <type_traits>

#include "../../detail/various.hpp"
#include "../../functional.hpp"
#include "../../intrinsics.hpp"
#include "../../types.hpp"

#include "../../block/block_load.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_store.hpp"

#include "../../device/device_scan_config.hpp"

#include "device_scan_common.hpp"
#include "lookback_scan_state.hpp"
#include "ordered_block_id.hpp"
#include "rocprim/intrinsics/thread.hpp"

BEGIN_ROCPRIM_NAMESPACE

// Single pass prefix scan was implemented based on:
// Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back.
// Technical Report NVR2016-001, NVIDIA Research. Mar. 2016.

namespace detail
{

// Helper functions for performing exclusive or inclusive
// block scan in single_scan.
template<bool Exclusive,
         class BlockScan,
         class T,
         unsigned int ItemsPerThread,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE auto single_scan_block_scan(T (&input)[ItemsPerThread],
                                                          T (&output)[ItemsPerThread],
                                                          T initial_value,
                                                          typename BlockScan::storage_type& storage,
                                                          BinaryFunction scan_op) ->
    typename std::enable_if<Exclusive>::type
{
    BlockScan().exclusive_scan(input, // input
                               output, // output
                               initial_value,
                               storage,
                               scan_op);
}

template<bool Exclusive,
         class BlockScan,
         class T,
         unsigned int ItemsPerThread,
         class BinaryFunction>
ROCPRIM_DEVICE ROCPRIM_INLINE auto single_scan_block_scan(T (&input)[ItemsPerThread],
                                                          T (&output)[ItemsPerThread],
                                                          T initial_value,
                                                          typename BlockScan::storage_type& storage,
                                                          BinaryFunction scan_op) ->
    typename std::enable_if<!Exclusive>::type
{
    (void)initial_value;
    BlockScan().inclusive_scan(input, // input
                               output, // output
                               storage,
                               scan_op);
}

template<bool Exclusive,
         class Config,
         class InputIterator,
         class OutputIterator,
         class BinaryFunction,
         class AccType,
         class LookbackScanState>
ROCPRIM_DEVICE ROCPRIM_FORCE_INLINE void
    lookback_scan_kernel_impl(InputIterator,
                              OutputIterator     output,
                              const size_t       size,
                              AccType            ,
                              BinaryFunction     scan_op,
                              LookbackScanState  scan_state,
                              const unsigned int,
                              AccType*                                 = nullptr,
                              AccType*                                 = nullptr,
                              bool                                     = false,
                              bool                                     = false)
{
    static_assert(std::is_same<AccType, typename LookbackScanState::value_type>::value,
                  "value_type of LookbackScanState must be result_type");
    static constexpr scan_config_params params = device_params<Config>();

    constexpr auto         block_size       = params.kernel_config.block_size;
    constexpr auto         items_per_thread = params.kernel_config.items_per_thread;
    constexpr unsigned int items_per_block  = block_size * items_per_thread;

    using block_store_type
        = ::rocprim::block_store<AccType, block_size, items_per_thread, params.block_store_method>;

    using lookback_scan_prefix_op_type
        = lookback_scan_prefix_op<AccType, BinaryFunction, LookbackScanState>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_store_type::storage_type store;
        rocprim::detail::raw_storage<AccType> prefix;
    } storage;

    const auto         flat_block_thread_id = ::rocprim::detail::block_thread_id<0>();
    const auto         flat_block_id        = ::rocprim::detail::block_id<0>();
    const unsigned int block_offset         = flat_block_id * items_per_block;
    const auto         valid_in_block
        = std::min(static_cast<unsigned int>(size - block_offset), items_per_block);

    // For input values
    AccType values[items_per_thread];
    for(unsigned i = 0; i < items_per_thread; ++i) {
        values[i] = AccType{0};
    }

    if(flat_block_id == 0)
    {
        if(flat_block_thread_id == 0)
        {
            scan_state.set_complete(0, AccType{0});
        }
    }
    else
    {
        // Scan of block values
        if (rocprim::warp_id(flat_block_thread_id) == 0) {
            auto prefix_op = lookback_scan_prefix_op_type(flat_block_id, scan_op, scan_state);
            AccType prefix = prefix_op(AccType{0});
            if (rocprim::lane_id() == 0)
                storage.prefix.get() = prefix;
        }
        ::rocprim::syncthreads();

        for(unsigned i = 0; i < items_per_thread; ++i) {
            values[i] = storage.prefix.get();
        }
    }
    ::rocprim::syncthreads(); // sync threads to reuse shared memory

    block_store_type().store(output + block_offset, values, valid_in_block, storage.store);
}

} // end of namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_SCAN_LOOKBACK_HPP_
