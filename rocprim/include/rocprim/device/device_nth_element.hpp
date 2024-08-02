// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_

#include "detail/device_nth_element.hpp"

#include "../detail/temp_storage.hpp"

#include "../config.hpp"

#include "config_types.hpp"
#include "device_nth_element_config.hpp"

#include <iostream>
#include <iterator>

#include <cstddef>
#include <cstdio>

BEGIN_ROCPRIM_NAMESPACE

/// \brief Parallel nth_element for device level.
///
/// \p nth_element function performs a device-wide nth_element,
/// this means that all number before the nth_element are smaller or equal
/// then the nth_element and after are larger or equal to the nth element
///
/// \par Overview
/// * The contents of the inputs are not altered by the function.
/// * Returns the required size of \p temporary_storage in \p storage_size
/// if \p temporary_storage in a null pointer.
/// * Accepts custom compare_functions for nth_element across the device.
/// * Does not work with hipGraph
///
/// \tparam KeysIterator - random-access iterator type of the input range. Must meet the
/// requirements of a C++ InputIterator concept. It can be a simple pointer type.
///
/// \param [in] temporary_storage - pointer to a device-accessible temporary storage. When
/// a null pointer is passed, the required allocation size (in bytes) is written to
/// \p storage_size and function returns without performing the sort operation.
/// \param [in,out] storage_size - reference to a size (in bytes) of \p temporary_storage.
/// \param [in] keys_input - pointer to the first element in the range to sort.
/// \param [out] keys_output - pointer to the first element in the output range.
/// \param [in] nth - The index of the nth_element in the input range.
/// \param [in] size - number of element in the input range.
/// \param [in] compare_function - binary operation function object that will be used for comparison.
/// The signature of the function should be equivalent to the following:
/// <tt>bool f(const T &a, const T &b);</tt>. The signature does not need to have
/// <tt>const &</tt>, but function object must not modify the objects passed to it.
/// The default value is \p BinaryFunction().
/// \param [in] stream - [optional] HIP stream object. Default is \p 0 (default stream).
/// \param [in] debug_synchronous - [optional] If true, synchronization after every kernel
/// launch is forced in order to check for errors. Default value is \p false.
///
/// \returns \p hipSuccess (\p 0) after successful sort; otherwise a HIP runtime error of
/// type \p hipError_t.
///
/// \par Example
/// \parblock
/// In this example a device-level nth_element is performed where input keys are
/// represented by an array of unsigned integers.
///
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// // Prepare input and output (declare pointers, allocate device memory etc.)
/// size_t input_size;          // e.g., 8
/// size_t nth;                 // e.g., 4
/// unsigned int * keys_input;  // e.g., [ 6, 3, 5, 4, 1, 8, 2, 7 ]
/// unsigned int * keys_output; // empty array of 8 elements
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
/// // Get required size of the temporary storage
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, nth, input_size
/// );
///
/// // allocate temporary storage
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// // perform nth_element
/// rocprim::nth_element(
///     temporary_storage_ptr, temporary_storage_size_bytes,
///     keys_input, keys_output, nth, input_size
/// );
/// // possible keys_output:   [ 1, 3, 4, 2, 5, 8, 7, 6 ]
/// \endcode
/// \endparblock
template<class Config = default_config,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<KeysIterator>::value_type>>
ROCPRIM_INLINE hipError_t nth_element(void*          temporary_storage,
                                      size_t&        storage_size,
                                      KeysIterator   keys_input,
                                      KeysIterator   keys_output,
                                      size_t         nth,
                                      size_t         size,
                                      BinaryFunction compare_function  = BinaryFunction(),
                                      hipStream_t    stream            = 0,
                                      bool           debug_synchronous = false)
{
    using config = detail::wrapped_nth_element_config<Config, Key>;
    detail::target_arch target_arch;
    hipError_t          result = host_target_arch(stream, target_arch);
    if(result != hipSuccess)
    {
        return result;
    }
    const detail::nth_element_config_params params
        = detail::dispatch_target_arch<config>(target_arch);

    const unsigned int num_buckets           = params.NumberOfBuckets;
    const unsigned int num_splitters         = num_buckets - 1;
    const unsigned int min_size              = num_buckets;
    const unsigned int num_items_per_threads = params.ItemsPerThread;
    const unsigned int num_threads_per_block = params.BlockSize;
    const unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;
    const unsigned int num_blocks = (size + num_items_per_block - 1) / num_items_per_block;

    Key*                             tree             = nullptr;
    size_t*                          buckets          = nullptr;
    detail::nth_element_data_type*   nth_element_data = nullptr;
    unsigned char*                   oracles          = nullptr;
    bool*                            equality_buckets = nullptr;
    detail::onesweep_lookback_state* lookback_states  = nullptr;

    Key* output = nullptr;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&tree, num_splitters),
            detail::temp_storage::ptr_aligned_array(&equality_buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&oracles, size),
            detail::temp_storage::ptr_aligned_array(&output, size),
            detail::temp_storage::ptr_aligned_array(&nth_element_data, 1),
            detail::temp_storage::ptr_aligned_array(&lookback_states, 3 * num_blocks)));

    if(partition_result != hipSuccess || temporary_storage == nullptr)
    {
        return partition_result;
    }

    if(size == 0 || nth >= size)
    {
        return hipSuccess;
    }

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << std::endl;
        std::cout << "num_buckets: " << num_buckets << std::endl;
        std::cout << "num_threads_per_block: " << num_threads_per_block << std::endl;
        std::cout << "num_blocks: " << num_blocks << std::endl;
        std::cout << "storage_size: " << storage_size << std::endl;
    }

    if(keys_input != keys_output)
    {
        hipError_t error
            = hipMemcpy(keys_output, keys_input, sizeof(Key) * size, hipMemcpyDeviceToDevice);
        if(error != hipSuccess)
        {
            return error;
        }
    }

    const unsigned int tree_depth = std::log2(num_buckets);
    detail::nth_element_keys_impl<config>(keys_output,
                                          output,
                                          tree,
                                          nth,
                                          size,
                                          buckets,
                                          equality_buckets,
                                          oracles,
                                          lookback_states,
                                          num_buckets,
                                          min_size,
                                          num_threads_per_block,
                                          num_items_per_threads,
                                          tree_depth,
                                          nth_element_data,
                                          compare_function,
                                          stream,
                                          debug_synchronous,
                                          0);

    return hipSuccess;
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_NTH_ELEMENT_HPP_