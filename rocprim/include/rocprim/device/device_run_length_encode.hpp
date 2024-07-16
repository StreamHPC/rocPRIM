// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
#define ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_

#include "device_reduce_by_key.hpp"
#include "device_run_length_encode_config.hpp"

#include "detail/device_config_helper.hpp"
#include "detail/device_run_length_encode.hpp"

#include "../common.hpp"
#include "../config.hpp"
#include "../detail/various.hpp"
#include "../iterator/constant_iterator.hpp"
#include "../type_traits.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <ios>
#include <iostream>
#include <iterator>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

namespace run_length_encode
{

template<typename Config,
         typename OffsetCountPairType,
         typename InputIterator,
         typename OffsetsOutputIterator,
         typename CountsOutputIterator,
         typename RunsCountOutputIterator,
         typename LookbackScanState>
ROCPRIM_KERNEL
    __launch_bounds__(Config::block_size)
void non_trivial_kernel(const InputIterator                  input,
                        const OffsetsOutputIterator          offsets_output,
                        const CountsOutputIterator           counts_output,
                        const RunsCountOutputIterator        runs_count_output,
                        const LookbackScanState              scan_state,
                        const ordered_block_id<unsigned int> ordered_tile_id,
                        const std::size_t                    number_of_tiles,
                        const std::size_t                    size)
{
    run_length_encode::non_trivial_kernel_impl<Config, OffsetCountPairType>(input,
                                                                            offsets_output,
                                                                            counts_output,
                                                                            runs_count_output,
                                                                            scan_state,
                                                                            ordered_tile_id,
                                                                            number_of_tiles,
                                                                            size);
}

template<class Config,
         class InputIterator,
         class OffsetsOutputIterator,
         class CountsOutputIterator,
         class RunsCountOutputIterator>
hipError_t run_length_encode_non_trivial_runs_impl(void*                   temporary_storage,
                                                   size_t&                 storage_size,
                                                   InputIterator           input,
                                                   const size_t            size,
                                                   OffsetsOutputIterator   offsets_output,
                                                   CountsOutputIterator    counts_output,
                                                   RunsCountOutputIterator runs_count_output,
                                                   const hipStream_t       stream,
                                                   const bool              debug_synchronous)
{
    using input_type  = ::rocprim::detail::value_type_t<InputIterator>;
    using offset_type = unsigned int;
    using count_type  = unsigned int;
    using offset_count_pair_type
        = run_length_encode::offset_count_pair_type_t<offset_type, count_type>; // accumulator_type

    using config = detail::default_or_custom_config<
        Config,
        reduce_by_key::default_config<ROCPRIM_TARGET_ARCH, input_type, offset_count_pair_type>>;

    using scan_state_type
        = ::rocprim::detail::lookback_scan_state<offset_count_pair_type, /*UseSleep=*/false>;
    using scan_state_with_sleep_type
        = ::rocprim::detail::lookback_scan_state<offset_count_pair_type, /*UseSleep=*/true>;

    using ordered_tile_id_type = detail::ordered_block_id<unsigned int>;

    constexpr unsigned int block_size      = config::block_size;
    constexpr unsigned int tiles_per_block = config::tiles_per_block;
    constexpr unsigned int items_per_tile  = block_size * config::items_per_thread;

    // Number of tiles in a single launch (or the only launch if it fits)
    const std::size_t number_of_tiles  = detail::ceiling_div(size, items_per_tile);
    const std::size_t number_of_blocks = detail::ceiling_div(number_of_tiles, tiles_per_block);

    // Calculate required temporary storage
    void*                          scan_state_storage;
    ordered_tile_id_type::id_type* ordered_bid_storage;

    detail::temp_storage::layout layout{};
    hipError_t result = scan_state_type::get_temp_storage_layout(number_of_tiles, stream, layout);
    if(result != hipSuccess)
    {
        return result;
    }

    result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            // This is valid even with scan_state_with_sleep_type
            detail::temp_storage::make_partition(&scan_state_storage, layout),
            detail::temp_storage::make_partition(&ordered_bid_storage,
                                                 ordered_tile_id_type::get_temp_storage_layout())));
    if(result != hipSuccess || temporary_storage == nullptr)
    {
        return result;
    }

    bool use_sleep;
    result = detail::is_sleep_scan_state_used(stream, use_sleep);
    if(result != hipSuccess)
    {
        return result;
    }

    scan_state_type scan_state{};
    result = scan_state_type::create(scan_state, scan_state_storage, number_of_tiles, stream);
    scan_state_with_sleep_type scan_state_with_sleep{};
    result = scan_state_with_sleep_type::create(scan_state_with_sleep,
                                                scan_state_storage,
                                                number_of_tiles,
                                                stream);

    if(result != hipSuccess)
    {
        return result;
    }

    auto with_scan_state
        = [use_sleep, scan_state, scan_state_with_sleep](auto&& func) mutable -> decltype(auto)
    {
        if(use_sleep)
        {
            return func(scan_state_with_sleep);
        }
        else
        {
            return func(scan_state);
        }
    };

    auto ordered_bid = ordered_tile_id_type::create(ordered_bid_storage);

    if(size == 0)
    {
        // Fill out runs_count_output with zero
        return rocprim::transform(rocprim::constant_iterator<std::size_t>(0),
                                  runs_count_output,
                                  1,
                                  rocprim::identity<std::size_t>{},
                                  stream,
                                  debug_synchronous);
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;
    if(debug_synchronous)
    {
        std::cout << "size:               " << size << '\n';
        std::cout << "block_size:         " << block_size << '\n';
        std::cout << "tiles_per_block:    " << tiles_per_block << '\n';
        std::cout << "number_of_tiles:    " << number_of_tiles << '\n';
        std::cout << "number_of_blocks:   " << number_of_blocks << '\n';
        std::cout << "items_per_tile:     " << items_per_tile << '\n';
        start = std::chrono::high_resolution_clock::now();
    }

    with_scan_state(
        [&](const auto scan_state)
        {
            const unsigned int block_size = ROCPRIM_DEFAULT_MAX_BLOCK_SIZE;
            const std::size_t  grid_size  = detail::ceiling_div(number_of_tiles, block_size);
            hipLaunchKernelGGL(init_lookback_scan_state_kernel,
                               dim3(grid_size),
                               dim3(block_size),
                               0,
                               stream,
                               scan_state,
                               number_of_tiles,
                               ordered_bid);
        });
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("init_lookback_scan_state_kernel",
                                                number_of_tiles,
                                                start);

    with_scan_state(
        [&](const auto scan_state)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    run_length_encode::non_trivial_kernel<config, offset_count_pair_type>),
                dim3(number_of_blocks),
                dim3(block_size),
                0,
                stream,
                input + 0,
                offsets_output,
                counts_output,
                runs_count_output,
                scan_state,
                ordered_bid,
                number_of_tiles,
                size);
        });
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("run_length_encode_non_trivial_runs kernel",
                                                size,
                                                start);

    return hipSuccess;
}
} // namespace run_length_encode
} // end detail namespace

/// \brief Parallel run-length encoding for device level.
///
/// run_length_encode function performs a device-wide run-length encoding of runs (groups)
/// of consecutive values. The first value of each run is copied to \p unique_output and
/// the length of the run is written to \p counts_output.
/// The total number of runs is written to \p runs_count_output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p runs_count_output must have at least 1 element.
/// * Ranges specified by \p unique_output and \p counts_output must have at least
/// <tt>*runs_count_output</tt> (i.e. the number of runs) elements.
///
/// \tparam Config - [optional] Configuration of the primitive, must be `default_config` or `run_length_encode_config`.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam UniqueOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam CountsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam RunsCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range of values.
/// \param [in] size - number of element in the input range.
/// \param [out] unique_output - iterator to the first element in the output range of unique values.
/// \param [out] counts_output - iterator to the first element in the output range of lenghts.
/// \param [out] runs_count_output - iterator to total number of runs.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level run-length encoding operation is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * input;                // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * unique_output;        // empty array of at least 4 elements
/// int * counts_output;        // empty array of at least 4 elements
/// int * runs_count_output;    // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::run_length_encode(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     unique_output, counts_output, runs_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform encoding
/// rocprim::run_length_encode(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     unique_output, counts_output, runs_count_output
/// );
/// // unique_output:     [1, 2, 10, 88]
/// // counts_output:     [3, 1,  3,  1]
/// // runs_count_output: [4]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class UniqueOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
hipError_t run_length_encode(void * temporary_storage,
                             size_t& storage_size,
                             InputIterator input,
                             unsigned int size,
                             UniqueOutputIterator unique_output,
                             CountsOutputIterator counts_output,
                             RunsCountOutputIterator runs_count_output,
                             hipStream_t stream = 0,
                             bool debug_synchronous = false)
{
    using input_type = typename std::iterator_traits<InputIterator>::value_type;
    using count_type = unsigned int;

    // run_length_encode does not use a tuned reduce by key config, as using a constant iterator
    //   instead of a device array has different performance characteristics
    using config = detail::default_or_custom_config<
        Config,
        run_length_encode_config<
            typename detail::default_reduce_by_key_config_base<input_type, count_type>::type,
            default_config>>;

    return ::rocprim::reduce_by_key<typename config::reduce_by_key>(
        temporary_storage, storage_size,
        input, make_constant_iterator<count_type>(1), size,
        unique_output, counts_output, runs_count_output,
        ::rocprim::plus<count_type>(), ::rocprim::equal_to<input_type>(),
        stream, debug_synchronous
    );
}

/// \brief Parallel run-length encoding of non-trivial runs for device level.
///
/// run_length_encode_non_trivial_runs function performs a device-wide run-length encoding of
/// non-trivial runs (groups) of consecutive values (groups of more than one element).
/// The offset of the first value of each non-trivial run is copied to \p offsets_output and
/// the length of the run (the count of elements) is written to \p counts_output.
/// The total number of non-trivial runs is written to \p runs_count_output.
///
/// \par Overview
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Range specified by \p input must have at least \p size elements.
/// * Range specified by \p runs_count_output must have at least 1 element.
/// * Ranges specified by \p offsets_output and \p counts_output must have at least
/// <tt>*runs_count_output</tt> (i.e. the number of non-trivial runs) elements.
///
/// \tparam Config - [optional] Configuration of the primitive, must be `default_config` or `run_length_encode_config`.
/// \tparam InputIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
/// \tparam OffsetsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam CountsOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
/// \tparam RunsCountOutputIterator - random-access iterator type of the output range. Must meet the
/// requirements of a C++ OutputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] input - iterator to the first element in the range of values.
/// \param [in] size - number of element in the input range.
/// \param [out] offsets_output - iterator to the first element in the output range of offsets.
/// \param [out] counts_output - iterator to the first element in the output range of lenghts.
/// \param [out] runs_count_output - iterator to total number of runs.
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful operation; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level run-length encoding of non-trivial runs is performed on an array of
/// integer values.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// int * input;                // e.g., [1, 1, 1, 2, 10, 10, 10, 88]
/// int * offsets_output;       // empty array of at least 2 elements
/// int * counts_output;        // empty array of at least 2 elements
/// int * runs_count_output;    // empty array of 1 element
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::run_length_encode_non_trivial_runs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     offsets_output, counts_output, runs_count_output
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform encoding
/// rocprim::run_length_encode_non_trivial_runs(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     input, input_size,
///     offsets_output, counts_output, runs_count_output
/// );
/// // offsets_output:    [0, 4]
/// // counts_output:     [3, 3]
/// // runs_count_output: [2]
/// \endcode
/// \endparblock
template<
    class Config = default_config,
    class InputIterator,
    class OffsetsOutputIterator,
    class CountsOutputIterator,
    class RunsCountOutputIterator
>
inline
hipError_t run_length_encode_non_trivial_runs(void * temporary_storage,
                                              size_t& storage_size,
                                              InputIterator input,
                                              unsigned int size,
                                              OffsetsOutputIterator offsets_output,
                                              CountsOutputIterator counts_output,
                                              RunsCountOutputIterator runs_count_output,
                                              hipStream_t stream = 0,
                                              bool debug_synchronous = false)
{
    return detail::run_length_encode::run_length_encode_non_trivial_runs_impl<Config>(
        temporary_storage,
        storage_size,
        input,
        size,
        offsets_output,
        counts_output,
        runs_count_output,
        stream,
        debug_synchronous);
}



/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_RUN_LENGTH_ENCODE_HPP_
