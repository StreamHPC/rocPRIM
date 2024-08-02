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

#ifndef ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_
#define ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_

#include "../../block/block_load.hpp"
#include "../../block/block_radix_rank.hpp"
#include "../../block/block_scan.hpp"
#include "../../block/block_sort.hpp"
#include "../../block/block_store.hpp"

#include "../../config.hpp"
#include "device_config_helper.hpp"

#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <iostream>
#include <rocprim/block/block_radix_rank.hpp>
#include <rocprim/config.hpp>
#include <rocprim/intrinsics/atomic.hpp>
#include <rocprim/intrinsics/thread.hpp>

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

#define ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR(name, size, start)                           \
    {                                                                                            \
        auto _error = hipGetLastError();                                                         \
        if(_error != hipSuccess)                                                                 \
            return _error;                                                                       \
        if(debug_synchronous)                                                                    \
        {                                                                                        \
            std::cout << name << "(" << size << ")";                                             \
            auto __error = hipStreamSynchronize(stream);                                         \
            if(__error != hipSuccess)                                                            \
                return __error;                                                                  \
            auto _end = std::chrono::high_resolution_clock::now();                               \
            auto _d   = std::chrono::duration_cast<std::chrono::duration<double>>(_end - start); \
            std::cout << " " << _d.count() * 1000 << " ms" << '\n';                              \
        }                                                                                        \
    }

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void
    kernel_block_sort(KeysIterator keys, const size_t size, BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int min_size = params.MinimumSize;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, min_size>;

    __shared__ typename block_sort_key::storage_type storage;

    size_t idx = threadIdx.x;
    Key    sample_buffer;

    if(idx < size)
    {
        sample_buffer = keys[idx];
    }

    block_sort_key().sort(sample_buffer, storage, size, compare_function);

    if(idx < size)
    {
        keys[idx] = sample_buffer;
    }
}

template<class config,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type>
ROCPRIM_KERNEL void kernel_copy_buckets(KeysIterator   keys,
                                        const size_t   size,
                                        size_t*        buckets_per_block_offsets,
                                        unsigned char* oracles,
                                        KeysIterator   output)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.NumberOfBuckets;
    constexpr unsigned int num_threads_per_block = params.BlockSize;
    constexpr unsigned int num_items_per_threads = params.ItemsPerThread;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;

    using block_load_bucket_t
        = rocprim::block_load<unsigned char, num_threads_per_block, num_items_per_threads>;
    using block_rank_t
        = rocprim::block_radix_rank<num_threads_per_block, 2, params.RadixRankAlgorithm>;
    using block_load_element_t
        = rocprim::block_load<Key, num_threads_per_block, num_items_per_threads>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_bucket_t::storage_type  load_bucket;
        typename block_rank_t::storage_type         rank;
        typename block_load_element_t::storage_type load_element;
    } storage;

    __shared__ unsigned int local_bucket_count[num_buckets];
    __shared__ size_t       buckets_block_offsets_shared[3];

    unsigned char buckets[num_items_per_threads];

    if(threadIdx.x < 3)
    {
        buckets_block_offsets_shared[threadIdx.x]
            = buckets_per_block_offsets[threadIdx.x * gridDim.x + blockIdx.x];
    }

    const size_t offset = blockIdx.x * num_items_per_block;

    if(offset + num_items_per_block <= size)
    {
        block_load_bucket_t().load(oracles + offset, buckets, storage.load_bucket);
    }
    else
    {
        const size_t valid = size - offset;
        block_load_bucket_t().load(oracles + offset, buckets, valid, 3, storage.load_bucket);
    }

    unsigned int ranks[num_items_per_threads];
    unsigned int digit_prefix[block_rank_t::digits_per_thread];
    unsigned int digit_counts[block_rank_t::digits_per_thread];
    block_rank_t().rank_keys(
        buckets,
        ranks,
        storage.rank,
        [](const int& key) { return key; },
        digit_prefix,
        digit_counts);

    if(threadIdx.x < 3)
    {
        local_bucket_count[threadIdx.x] = digit_prefix[0];
    }

    rocprim::syncthreads();

    Key elements[num_items_per_threads];
    if(offset + num_items_per_block <= size)
    {
        block_load_element_t().load(keys + offset, elements, storage.load_element);
    }
    else
    {
        block_load_element_t().load(keys + offset,
                                    elements,
                                    size - offset,
                                    0,
                                    storage.load_element);
    }

    const size_t thread_id = (threadIdx.x * num_items_per_threads) + offset;

    ROCPRIM_UNROLL
    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        const size_t idx = item + thread_id;
        if(idx < size)
        {
            const auto bucket = buckets[item];
            size_t     index  = buckets_block_offsets_shared[bucket];
            index += (ranks[item] - local_bucket_count[bucket]);
            output[index] = elements[item];
        }
    }
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_find_splitters(KeysIterator   keys,
                                          KeysIterator   tree,
                                          bool*          equality_buckets,
                                          const size_t   size,
                                          BinaryFunction compare_function)
{
    constexpr nth_element_config_params params        = device_params<config>();
    constexpr unsigned int              num_splitters = params.NumberOfBuckets - 1;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_sort_key = rocprim::block_sort<Key, num_splitters>;

    __shared__ typename block_sort_key::storage_type storage;

    const auto offset = size / num_splitters;
    auto       idx    = threadIdx.x;

    auto sample_buffer = keys[offset + offset * idx];
    block_sort_key().sort(sample_buffer, storage, compare_function);

    tree[idx] = sample_buffer;

    rocprim::syncthreads();

    bool equality_bucket = false;
    if(idx > 0)
    {
        equality_bucket
            = tree[idx - 1] == sample_buffer
              && (idx == num_splitters - 1 || compare_function(sample_buffer, tree[idx + 1]));
    }

    equality_buckets[idx] = equality_bucket;
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_count_bucket_sizes(KeysIterator   keys,
                                              KeysIterator   tree,
                                              const size_t   size,
                                              size_t*        buckets,
                                              bool*          equality_buckets,
                                              const size_t   tree_depth,
                                              BinaryFunction compare_function)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.NumberOfBuckets;
    constexpr unsigned int num_threads_per_block = params.BlockSize;
    constexpr unsigned int num_items_per_threads = params.ItemsPerThread;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key = rocprim::
        block_load<Key, num_threads_per_block, num_items_per_threads>;

    struct storage_type_
    {
        Key search_tree[num_splitters];
    };
    using storage_type = detail::raw_storage<storage_type_>;

    __shared__ storage_type                          storage;
    __shared__ typename block_load_key::storage_type key_load_storage;
    __shared__ unsigned int                          shared_buckets[num_buckets];

    if(threadIdx.x < num_buckets)
    {
        shared_buckets[threadIdx.x] = 0;
    }

    storage_type_& storage_ = storage.get();
    if(threadIdx.x < num_splitters)
    {
        storage_.search_tree[threadIdx.x] = tree[threadIdx.x];
    }

    Key          elements[num_items_per_threads];
    const size_t offset = blockIdx.x * num_items_per_block;

    if(offset + num_items_per_block < size)
    {
        block_load_key().load(keys + offset, elements, key_load_storage);
    }
    else
    {
        block_load_key().load(keys + offset, elements, size - offset, key_load_storage);
    }

    rocprim::syncthreads();

    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        auto idx = offset + threadIdx.x * num_items_per_threads + item;
        if(idx < size)
        {
            auto         element = elements[item];
            unsigned int bucket  = num_splitters / 2;
            auto         diff    = num_buckets / 2;
            for(unsigned int i = 0; i < tree_depth - 1; i++)
            {
                diff = diff / 2;
                bucket += compare_function(element, storage_.search_tree[bucket]) ? -diff : diff;
            }

            if(!compare_function(element, storage_.search_tree[bucket]))
            {
                bucket++;
            }

            if(bucket > 0 && equality_buckets[bucket - 1]
               && element == storage_.search_tree[bucket - 1])
            {
                bucket = bucket - 1;
            }

            detail::atomic_add(&shared_buckets[bucket], 1);
        }
    }

    rocprim::syncthreads();

    if(threadIdx.x < num_buckets)
    {
        detail::atomic_add(&buckets[threadIdx.x], shared_buckets[threadIdx.x]);
    }
}

template<class config, class KeysIterator, class BinaryFunction>
ROCPRIM_KERNEL void kernel_store_buckets(KeysIterator   keys,
                                         KeysIterator   tree,
                                         const size_t   size,
                                         size_t*        buckets_per_block_offsets,
                                         bool*          equality_buckets,
                                         unsigned char* oracles,
                                         BinaryFunction compare_function,
                                         size_t*        nth_element_data)
{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets           = params.NumberOfBuckets;
    constexpr unsigned int num_threads_per_block = params.BlockSize;
    constexpr unsigned int num_items_per_threads = params.ItemsPerThread;
    constexpr unsigned int num_splitters         = num_buckets - 1;
    constexpr unsigned int num_items_per_block   = num_threads_per_block * num_items_per_threads;

    using Key = typename std::iterator_traits<KeysIterator>::value_type;

    using block_load_key = rocprim::
        block_load<Key, num_threads_per_block, num_items_per_threads>;
    using block_store_oracle
        = rocprim::block_store<unsigned char, num_threads_per_block, num_items_per_threads>;

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_load_key::storage_type     load;
        typename block_store_oracle::storage_type store;
    } storage;

    struct storage_type_
    {
        Key           search_tree[2];
        unsigned char equality_buckets[2];
    };
    using storage_type = detail::raw_storage<storage_type_>;

    __shared__ storage_type raw_storage;
    __shared__ size_t       shared_buckets[3];

    if(threadIdx.x < 3)
    {
        shared_buckets[threadIdx.x] = 0;
    }

    storage_type_& storage_ = raw_storage.get();

    const auto nth_element = nth_element_data[3];
    if(threadIdx.x < 2)
    {
        if(threadIdx.x < 1 || nth_element < num_splitters)
        {
            const auto index = (nth_element > 0 ? (nth_element - 1) + threadIdx.x : 0);
            storage_.search_tree[threadIdx.x]      = tree[index];
            storage_.equality_buckets[threadIdx.x] = equality_buckets[index];
        }
    }

    Key          elements[num_items_per_threads];
    const size_t offset = blockIdx.x * num_items_per_block;

    if(offset + num_items_per_block < size)
    {
        block_load_key().load(keys + offset, elements, storage.load);
    }
    else
    {
        block_load_key().load(keys + offset, elements, size - offset, storage.load);
    }

    rocprim::syncthreads();

    const size_t  thread_offset = offset + threadIdx.x * num_items_per_threads;
    unsigned char local_oracles[num_items_per_threads];

    for(size_t item = 0; item < num_items_per_threads; item++)
    {
        const size_t idx = thread_offset + item;
        if(idx < size)
        {
            Key           element = elements[item];
            unsigned char bucket  = 0;

            if(nth_element > 0)
            {
                if(storage_.equality_buckets[0])
                {
                    bucket += !compare_function(element, storage_.search_tree[0])
                              && !(element == storage_.search_tree[0]);
                }
                else
                {
                    bucket += !compare_function(element, storage_.search_tree[0]);
                }
            }

            if(nth_element < num_splitters)
            {
                if(storage_.equality_buckets[1])
                {
                    bucket += !compare_function(element, storage_.search_tree[1])
                              && !(element == storage_.search_tree[1]);
                }
                else
                {
                    bucket += !compare_function(element, storage_.search_tree[1]);
                }
            }

            local_oracles[item] = bucket;

            detail::atomic_add(&shared_buckets[bucket], 1);
        }
    }

    if(offset + num_items_per_block < size)
    {
        block_store_oracle().store(oracles + offset, local_oracles, storage.store);
    }
    else
    {
        block_store_oracle().store(oracles + offset, local_oracles, size - offset, storage.store);
    }

    rocprim::syncthreads();

    if(threadIdx.x < 3)
    {
        buckets_per_block_offsets[threadIdx.x * gridDim.x + blockIdx.x]
            = shared_buckets[threadIdx.x];
    }
}

ROCPRIM_KERNEL void kernel_calc_block_offset(size_t*      buckets_per_block_offsets,
                                             size_t*      nth_element_data,
                                             const size_t num_blocks)

{
    size_t bucket_offset = 0;

    if(threadIdx.x > 0)
    {
        const size_t nth_bucket_offset = nth_element_data[0];
        bucket_offset += nth_bucket_offset;
    }

    const bool equality_bucket = nth_element_data[3];
    if((threadIdx.x > 0 && !equality_bucket) || threadIdx.x > 1)
    {
        const size_t nth_bucket_size = nth_element_data[1];
        bucket_offset += nth_bucket_size;
    }

    size_t       counter      = bucket_offset;
    const size_t block_offset = threadIdx.x * num_blocks;
    const size_t max          = block_offset + num_blocks;

    ROCPRIM_UNROLL
    for(size_t idx = block_offset; idx < max; idx++)
    {
        size_t offset                  = buckets_per_block_offsets[idx];
        buckets_per_block_offsets[idx] = counter;
        counter += offset;
    }
}

template<class config>
ROCPRIM_KERNEL void kernel_find_nth_element_bucket(size_t*            buckets_per_block_offsets,
                                                   size_t*            buckets,
                                                   size_t*            nth_element_data,
                                                   bool*              equality_buckets,
                                                   const size_t       rank,
                                                   const unsigned int num_blocks)

{
    constexpr nth_element_config_params params = device_params<config>();

    constexpr unsigned int num_buckets = params.NumberOfBuckets;

    using block_scan_offsets  = rocprim::block_scan<size_t, num_buckets>;
    using block_scan_find_nth = rocprim::block_scan<size_t, num_buckets>;

    __shared__ size_t bucket_offsets[num_buckets];

    ROCPRIM_SHARED_MEMORY union
    {
        typename block_scan_offsets::storage_type  bucket_offset;
        typename block_scan_find_nth::storage_type find_nth;
    } storage;

    size_t bucket_offset;
    size_t bucket_size = buckets[threadIdx.x];
    block_scan_offsets().exclusive_scan(bucket_size, bucket_offset, 0, storage.bucket_offset);

    bucket_offsets[threadIdx.x] = bucket_offset;

    rocprim::syncthreads();

    size_t num_buckets_before;

    // Find the data of the nth element
    bool in_nth = bucket_offset <= rank;

    block_scan_find_nth().inclusive_scan(in_nth, num_buckets_before, storage.find_nth);

    if(threadIdx.x == (num_buckets - 1))
    {
        auto nth_element    = num_buckets_before - 1;
        nth_element_data[0] = bucket_offsets[nth_element];
        nth_element_data[1] = buckets[nth_element];
        nth_element_data[2] = equality_buckets[nth_element];
        nth_element_data[3] = nth_element;
    }
}

template<class config,
         class KeysIterator,
         class Key = typename std::iterator_traits<KeysIterator>::value_type,
         class BinaryFunction>
ROCPRIM_INLINE hipError_t nth_element_keys_impl(KeysIterator       keys,
                                                KeysIterator       output,
                                                KeysIterator       tree,
                                                const size_t       rank,
                                                const size_t       size,
                                                size_t*            buckets,
                                                size_t*            buckets_per_block_offsets,
                                                bool*              equality_buckets,
                                                unsigned char*     oracles,
                                                const unsigned int num_buckets,
                                                const unsigned int min_size,
                                                const unsigned int num_threads_per_block,
                                                const unsigned int num_items_per_threads,
                                                const unsigned int tree_depth,
                                                size_t*            nth_element_data,
                                                BinaryFunction     compare_function,
                                                hipStream_t        stream,
                                                bool               debug_synchronous,
                                                const size_t       recursion)
{
    const unsigned int num_splitters       = num_buckets - 1;
    const unsigned int num_items_per_block = num_threads_per_block * num_items_per_threads;
    const unsigned int num_blocks          = (size + num_items_per_block - 1) / num_items_per_block;

    if(debug_synchronous)
    {
        std::cout << "-----" << '\n';
        std::cout << "size: " << size << std::endl;
        std::cout << "rank: " << rank << std::endl;
        std::cout << "recursion level: " << recursion << std::endl;
    }

    // Start point for time measurements
    std::chrono::high_resolution_clock::time_point start;

    if(size < min_size)
    {
        if(debug_synchronous)
        {
            start = std::chrono::high_resolution_clock::now();
        }
        kernel_block_sort<config><<<1, min_size, 0, stream>>>(keys, size, compare_function);
        ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_block_sort", size, start);
        return hipSuccess;
    }

    hipError_t error = hipMemsetAsync(buckets, 0, sizeof(size_t) * num_buckets, stream);

    if(error != hipSuccess)
    {
        return error;
    }

    error = hipMemsetAsync(equality_buckets, 0, sizeof(bool) * num_buckets, stream);

    if(error != hipSuccess)
    {
        return error;
    }

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    // Currently launches power of 2 minus 1 threads
    kernel_find_splitters<config>
        <<<1, num_splitters, 0, stream>>>(keys, tree, equality_buckets, size, compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_splitters", size, start);
    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_count_bucket_sizes<config>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                           tree,
                                                           size,
                                                           buckets,
                                                           equality_buckets,
                                                           tree_depth,
                                                           compare_function);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_count_bucket_sizes", size, start);
    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_find_nth_element_bucket<config><<<1, num_buckets, 0, stream>>>(buckets_per_block_offsets,
                                                                          buckets,
                                                                          nth_element_data,
                                                                          equality_buckets,
                                                                          rank,
                                                                          num_blocks);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_find_nth_element_bucket", size, start);

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_store_buckets<config>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                           tree,
                                                           size,
                                                           buckets_per_block_offsets,
                                                           equality_buckets,
                                                           oracles,
                                                           compare_function,
                                                           nth_element_data);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_store_buckets", size, start);

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_calc_block_offset<<<1, 3, 0, stream>>>(buckets_per_block_offsets,
                                                  nth_element_data,
                                                  num_blocks);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_calc_block_offset", size, start);

    if(debug_synchronous)
    {
        start = std::chrono::high_resolution_clock::now();
    }
    kernel_copy_buckets<config>
        <<<num_blocks, num_threads_per_block, 0, stream>>>(keys,
                                                           size,
                                                           buckets_per_block_offsets,
                                                           oracles,
                                                           output);
    ROCPRIM_DETAIL_HIP_SYNC_AND_RETURN_ON_ERROR("kernel_copy_buckets", size, start);

    error = hipMemcpyAsync(keys, output, sizeof(Key) * size, hipMemcpyDeviceToDevice);

    size_t h_nth_element_data[3];
    error = hipMemcpyAsync(&h_nth_element_data,
                           nth_element_data,
                           3 * sizeof(size_t),
                           hipMemcpyDeviceToHost);

    hipDeviceSynchronize();

    if(error != hipSuccess)
    {
        return error;
    }

    size_t offset          = h_nth_element_data[0];
    size_t bucket_size     = h_nth_element_data[1];
    bool   equality_bucket = h_nth_element_data[2];

    if(equality_bucket)
    {
        return hipSuccess;
    }

    return nth_element_keys_impl<config>(keys + offset,
                                         output,
                                         tree,
                                         rank - offset,
                                         bucket_size,
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
                                         recursion + 1);
}
} // namespace detail

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DETAIL_DEVICE_NTH_ELEMENT_HPP_