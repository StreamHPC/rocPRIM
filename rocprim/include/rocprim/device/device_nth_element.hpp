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
    const detail::nth_element_config_params params = detail::dispatch_target_arch<config>(target_arch);

    const unsigned int num_buckets           = params.NumberOfBuckets;
    const unsigned int num_splitters         = num_buckets - 1;
    const unsigned int min_size              = num_buckets;
    const unsigned int num_items_per_threads = params.ItemsPerThread;
    const unsigned int num_threads_per_block = params.BlockSize;
    const unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;
    const unsigned int num_blocks            = (size + num_items_per_block - 1) / num_items_per_block;

    Key*           tree                      = nullptr;
    size_t*        buckets                   = nullptr;
    size_t*        buckets_per_block_offsets = nullptr;
    size_t*        nth_element_data          = nullptr;
    unsigned char* oracles                   = nullptr;
    bool*          equality_buckets          = nullptr;

    Key* output = nullptr;

    const hipError_t partition_result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&tree, num_splitters),
            detail::temp_storage::ptr_aligned_array(&equality_buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&buckets, num_buckets),
            detail::temp_storage::ptr_aligned_array(&buckets_per_block_offsets, 3 * num_blocks),
            detail::temp_storage::ptr_aligned_array(&oracles, size),
            detail::temp_storage::ptr_aligned_array(&output, size),
            detail::temp_storage::ptr_aligned_array(&nth_element_data, 4)));

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
                                  buckets_per_block_offsets,
                                  equality_buckets,
                                  oracles,
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