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

#ifndef ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
#define ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_

#include "../config.hpp"
#include "../functional.hpp"

#include "../detail/temp_storage.hpp"
#include "../detail/various.hpp"

#include "../block/block_store.hpp"
#include "../device/config_types.hpp"
#include "../device/device_merge_inplace_config.hpp"
#include "../intrinsics/bit.hpp"
#include "../intrinsics/thread.hpp"
#include "../thread/thread_search.hpp"

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup devicemodule
/// @{

namespace detail
{

/// \brief implementation detail of merge inplace
template<size_t GlobalMergeBlockSize,
         size_t BlockMergeBlockSize,
         size_t BlockMergeIPT,
         typename IteratorT,
         typename OffsetT,
         typename BinaryFunction>
struct merge_inplace_impl
{
    using iterator_t = IteratorT;
    using value_t    = typename std::iterator_traits<iterator_t>::value_type;
    using offset_t   = OffsetT;

    static constexpr size_t global_merge_block_size      = GlobalMergeBlockSize;
    static constexpr size_t block_merge_block_size       = BlockMergeBlockSize;
    static constexpr size_t block_merge_items_per_thread = BlockMergeIPT;
    static constexpr size_t block_merge_items_per_block
        = block_merge_block_size * block_merge_items_per_thread;

    static constexpr offset_t no_split = -1;

    struct pivot_t
    {
        // rocprim::merge_path_search uses '.x' and '.y', but that isn't very descriptive.
        // so we union it with more descriptive names
        union
        {
            offset_t left;
            offset_t x;
        };
        union
        {
            offset_t right;
            offset_t y;
        };

        ROCPRIM_DEVICE pivot_t& offset(offset_t a, offset_t b)
        {
            left += a;
            right += b;
            return *this;
        }
    };

    /// \brief describes two ranges [begin, split) and [split, end)
    struct work_t
    {
        offset_t begin;
        offset_t split;
        offset_t end;

        ROCPRIM_DEVICE ROCPRIM_INLINE constexpr bool is_valid() const
        {
            return begin <= split && split <= end;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE constexpr bool has_work() const
        {
            return begin < split && split < end;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE offset_t total_size() const
        {
            return end - begin;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE offset_t left_size() const
        {
            return split - begin;
        }

        ROCPRIM_DEVICE ROCPRIM_INLINE offset_t right_size() const
        {
            return end - split;
        }

        static constexpr ROCPRIM_DEVICE ROCPRIM_INLINE work_t invalid_work()
        {
            return work_t{0, no_split, 0};
        }
    };

    /// \brief finds the `work_t` and its id by descending the binary tree `work_tree`.
    ROCPRIM_DEVICE ROCPRIM_INLINE work_t reconstruct_work(offset_t  worker_global_id,
                                                          offset_t* work_tree,
                                                          work_t    work,
                                                          uint32_t& work_id,
                                                          uint32_t  subdivisions)
    {
        work_id    = 1;
        work.split = work_tree[work_id];

        // we need an upper bound since another thread may have already written the next level
        for(uint32_t i = 0; i < subdivisions; ++i)
        {
            // move to next layer in binary tree
            work_id <<= 1;

            // check which side of the binary tree we descend
            if(worker_global_id >= work.split)
            {
                // worker id is right of split
                work.begin = work.split;
                work_id |= 1;
            }
            else
            {
                work.end = work.split;
            }
            work.split = work_tree[work_id];

            // early exit if we encounter a leaf!
            if(work.split == no_split)
                break;
        }
        return work;
    }

    /// \brief Reconstructs `work_t` from an index by traveling up the tree to the root.
    /// If `find_split` is `true`, also try to first find the first id that is valid
    /// while discarding duplicates.
    template<bool find_split = false>
    ROCPRIM_DEVICE ROCPRIM_INLINE static work_t reconstruct_work_from_id(work_t    work,
                                                                         offset_t* work_tree,
                                                                         offset_t  work_id,
                                                                         int       global_divisions)
    {
        bool need_begin = true;
        bool need_end   = true;
        work.split      = work_tree[work_id];

        for(int i = 0; i < global_divisions && (need_begin || need_end); ++i)
        {
            // odd ids are right branches
            bool is_right = work_id & 1;

            // if this isn't a split, move a layer up
            if(find_split && work.split == no_split)
            {
                // if the sibling is also not a split, prune the right branch
                if(is_right && work_tree[work_id & ~1] == no_split)
                    return work_t::invalid_work();

                work_id >>= 1;
                work.split = work_tree[work_id];
                continue;
            }

            // move to the next layer
            work_id >>= 1;

            // if the parent is a leaf do not update our internal state
            if(work_tree[work_id] == no_split)
            {
                // cut off the right branch and let the left branch inherit the parent
                if(is_right)
                    return work_t::invalid_work();
                continue;
            }

            // if we're the right child, then the parent split is the left bound
            if(is_right && work.begin == 0)
            {
                work.begin = work_tree[work_id];
                need_begin = false;
            }

            // if we're the left child, then the parent split is the right bound
            if(!is_right && need_end)
            {
                work.end = work_tree[work_id];
                need_end = false;
            }
        }

        return work;
    }

    using block_merge_block_store
        = block_store<value_t, block_merge_block_size, block_merge_items_per_thread>;

    static auto get_num_global_divisions(size_t left_size, size_t right_size)
    {
        const offset_t max_size = max(left_size, right_size);
        const int32_t  set_bits = std::numeric_limits<size_t>::digits - clz(max_size);

        // compute 2 + ceil(log_2(max(left, right))) - log_2(items_per_thread)
        return max(2, 2 + set_bits - Log2<block_merge_items_per_block>::VALUE);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void merge_inplace_agent(cooperative_groups::grid_group   grid,
                                                           cooperative_groups::thread_block block,
                                                           offset_t*      work_tree,
                                                           pivot_t*       pivot_heap,
                                                           offset_t*      scratch,
                                                           iterator_t     data,
                                                           offset_t       left_work_size,
                                                           offset_t       right_work_size,
                                                           BinaryFunction compare_function)
    {
        const uint32_t block_thread_id   = block.thread_rank();
        const uint32_t grid_thread_id    = grid.thread_rank();
        const uint32_t grid_thread_count = grid.size();

        const offset_t total_work_size = left_work_size + right_work_size;

        offset_t* global_work_granularity = &scratch[0];
        offset_t* global_division_counter = &scratch[1];

        __shared__ offset_t block_work_granularity;

        const work_t initial_work = work_t{0, left_work_size, total_work_size};

        // put first item on the heap, do it for each block since grid sync is more expensive
        if(block_thread_id == 0)
            work_tree[1] = initial_work.split;

        // dependent on first item on work heap
        block.sync();

        offset_t thread_work_granularity = total_work_size;

        // keep track of total number of subdivisions
        uint32_t subdivisions = 0;
        while(thread_work_granularity > block_merge_items_per_block)
        {
            // domain of the work ids
            const uint32_t work_id_begin = 1 << subdivisions;
            const uint32_t work_id_end   = work_id_begin * 2;

            // find and elect pivot candidates
            for(uint32_t work_id = work_id_begin + grid_thread_id; work_id < work_id_end;
                work_id += grid_thread_count)
            {
                const work_t work
                    = reconstruct_work_from_id(initial_work, work_tree, work_id, subdivisions);

                if(!work.is_valid() || work.total_size() <= block_merge_items_per_block)
                    continue;

                pivot_t pivot;
                rocprim::merge_path_search(work.total_size() / 2,
                                           data + work.begin,
                                           data + work.split,
                                           work.left_size(),
                                           work.right_size(),
                                           pivot,
                                           compare_function);

                pivot_heap[work_id] = pivot.offset(work.begin, work.split);
            }

            // elected pivot must be observable by all threads
            grid.sync();

            if(block_thread_id == 0)
                block_work_granularity = 0;

            // we can clear out global work since we are after the grid sinc after use
            if(grid_thread_id == 0)
                *global_work_granularity = 0;

            // rotate with double reverse:
            // initial : a0 a1 a2 a3 b0 b1 b2
            // step 1  : a3 a2 a1 a0 b2 b1 b0 // reverse a and b array
            // step 2  : b0 b1 b2 a0 a1 a2 a3 // reverse entire array

            // first reverse to start rotation
            // we can use `floor(N / 2)`  number of workers for a rotation of total size N since
            // each worker will swap 2 items
            for(offset_t worker_global_id = grid_thread_id * 2; worker_global_id < total_work_size;
                worker_global_id += grid_thread_count * 2)
            {
                uint32_t     work_id = 1;
                const work_t work    = reconstruct_work(worker_global_id,
                                                     work_tree,
                                                     initial_work,
                                                     work_id,
                                                     subdivisions);

                if(!work.has_work() || work.total_size() <= block_merge_items_per_block)
                    continue;

                const offset_t work_offset = (worker_global_id - work.begin) / 2;
                const pivot_t  pivot       = pivot_heap[work_id];

                if(work_offset >= (pivot.right - pivot.left) / 2)
                    continue;

                const offset_t mid_l = (pivot.left + work.split) / 2 - pivot.left;

                // reverse the left and right array separately
                const bool is_left = work_offset < mid_l;

                const offset_t from           = is_left ? pivot.left : work.split;
                const offset_t to             = is_left ? work.split : pivot.right;
                const offset_t reverse_offset = is_left ? work_offset : work_offset - mid_l;

                const offset_t a = from + reverse_offset;
                const offset_t b = to - reverse_offset - 1;

                if(a != b)
                    rocprim::swap(data[a], data[b]);
            }

            // first reverse must be observable by all threads
            grid.sync();

            // second reverse step to complete rotation
            for(offset_t worker_global_id = grid_thread_id * 2; worker_global_id < total_work_size;
                worker_global_id += grid_thread_count * 2)
            {
                uint32_t     work_id = 1;
                const work_t work    = reconstruct_work(worker_global_id,
                                                     work_tree,
                                                     initial_work,
                                                     work_id,
                                                     subdivisions);

                if(!work.has_work() || work.total_size() <= block_merge_items_per_block)
                    continue;

                const offset_t work_offset = (worker_global_id - work.begin) / 2;
                const pivot_t  pivot       = pivot_heap[work_id];

                if(work_offset >= (pivot.right - pivot.left) / 2)
                    continue;

                const auto a = pivot.left + work_offset;
                const auto b = pivot.right - work_offset - 1;

                if(a != b)
                    rocprim::swap(data[a], data[b]);
            }

            // rotate must be completed before modifying work heap
            grid.sync();

            thread_work_granularity = 0;

            // enqueue future work
            for(uint32_t work_id = work_id_begin + grid_thread_id; work_id < work_id_end;
                work_id += grid_thread_count)
            {
                const work_t work
                    = reconstruct_work_from_id(initial_work, work_tree, work_id, subdivisions);

                // default splits:
                offset_t new_split   = work.split;
                offset_t left_split  = no_split;
                offset_t right_split = no_split;

                // if this node isn't a leaf and the work should not be done in the block level merge
                if(work.is_valid() && work.total_size() > block_merge_items_per_block)
                {
                    const pivot_t pivot = pivot_heap[work_id];
                    if(!(pivot.left == work.split && pivot.right == work.end))
                    {
                        // the pivots describe the child work, but we have to adjust
                        // the work's split since we rotated around that value
                        new_split   = pivot.left + pivot.right - work.split;
                        left_split  = pivot.left;
                        right_split = pivot.right;
                    }

                    const offset_t left_size  = left_split == no_split ? 0 : new_split - work.begin;
                    const offset_t right_size = right_split == no_split ? 0 : work.end - new_split;

                    // update the thread work granularity
                    thread_work_granularity
                        = max(thread_work_granularity, max(left_size, right_size));

                    if(left_size == 0)
                        left_split = no_split;
                    if(right_size == 0)
                        right_split = no_split;
                }

                // write offset for the descendents
                const uint32_t child_work_id = work_id << 1;

                // write descendents to global memory
                work_tree[work_id]           = new_split;
                work_tree[child_work_id]     = left_split;
                work_tree[child_work_id | 1] = right_split;
            }

            // update block work granularity over shared memory
            atomicMax(&block_work_granularity, thread_work_granularity);

            // shared work granularity must be visible by block
            block.sync();

            // update grid work granularity over global memory
            if(block_thread_id == 0)
                atomicMax(global_work_granularity, block_work_granularity);

            // work heap modification must be visible before modifying pivot heap
            grid.sync();

            thread_work_granularity = *global_work_granularity;

            subdivisions++;
        }

        if(grid_thread_id == 0)
        {
            *global_division_counter = subdivisions;
        }
    }

    static __global__ void merge_inplace_kernel(offset_t*      work_storage,
                                                pivot_t*       pivot_storage,
                                                offset_t*      scratch_storage,
                                                iterator_t     data,
                                                size_t         left_size,
                                                size_t         right_size,
                                                BinaryFunction compare_function)
    {
        cooperative_groups::grid_group   grid  = cooperative_groups::this_grid();
        cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

        merge_inplace_impl{}.merge_inplace_agent(grid,
                                                 block,
                                                 work_storage,
                                                 pivot_storage,
                                                 scratch_storage,
                                                 data,
                                                 left_size,
                                                 right_size,
                                                 compare_function);
    }

    static __global__ void block_merge_kernel(iterator_t     data,
                                              size_t         num_items,
                                              BinaryFunction compare_function,
                                              offset_t*      work_tree,
                                              offset_t*      scratch_storage)
    {
        cooperative_groups::thread_block block = cooperative_groups::this_thread_block();

        const uint32_t grid_size       = rocprim::detail::grid_size<0>();
        const uint32_t block_id        = block.group_index().x;
        const uint32_t block_thread_id = block.thread_rank();
        const uint32_t subdivisions    = scratch_storage[1];

        work_t initial_work{0, no_split, num_items};

        // domain of the work ids
        const uint32_t work_id_begin = 1 << subdivisions;
        const uint32_t work_id_end   = work_id_begin * 2;

        value_t thread_data[block_merge_items_per_thread];

        // grid stride over the work ids
        for(uint32_t work_id = work_id_begin + block_id; work_id < work_id_end;
            work_id += grid_size)
        {
            const work_t work
                = reconstruct_work_from_id<true>(initial_work, work_tree, work_id, subdivisions);

            bool has_work = work.has_work() && work.total_size() <= block_merge_items_per_block;

            if(has_work)
            {
                // divide work over threads via merge path
                const offset_t diagonal = block_merge_items_per_thread * block_thread_id;

                pivot_t pivot;
                rocprim::merge_path_search(diagonal,
                                           data + work.begin,
                                           data + work.split,
                                           work.left_size(),
                                           work.right_size(),
                                           pivot,
                                           compare_function);
                pivot.offset(work.begin, work.split);

                // serial merge
                ROCPRIM_UNROLL
                for(uint32_t i = 0; i < block_merge_items_per_thread; ++i)
                {
                    if(block_merge_items_per_thread * block_thread_id + i >= work.total_size())
                        continue;

                    bool take_left = (pivot.right >= work.end)
                                     || ((pivot.left < work.split)
                                         && !compare_function(data[pivot.right], data[pivot.left]));

                    if(take_left)
                    {
                        thread_data[i] = data[pivot.left];
                        ++pivot.left;
                    }
                    else
                    {
                        thread_data[i] = data[pivot.right];
                        ++pivot.right;
                    }
                }

                // there are no partial blocks working on this, so a
                // block sync in this conditional can be done safely
                block.sync();

                block_merge_block_store{}.store(data + work.begin, thread_data, work.total_size());
            }
        }
    }
};

} // namespace detail

/// \brief Parallel merge inplace primitive for device level.
///
/// The `merge_inplace` function performs a device-wide merge in place. It merges two ordered sets
/// of input values based on a comparison function using significantly less temporary storage
/// compared to `merge`.
///
/// \warning This functions prioritizes temporary storage over speed. In most cases using `merge`
/// and a device copy is significantly faster.
///
/// \par Overview
/// * The function can write intermediate values to the data array while the algorithm is running.
/// * Returns the required size of `temporary_storage` in `storage_size` if `temporary_storage` is a
/// null pointer.
/// * Accepts a custom `compare_function`.
///
/// \tparam Config Configuration of the primitive, must be `default_config` or `merge_inplace_config`.
/// \tparam Iterator Random access iterator type for the input and output range. Must meet the
/// requirements of `std::random_access_iterator`.
/// \tparam BinaryFunction Binary function type that is used for the comparison.
///
/// \param [in] temporary_storage Pointer to a device-accessible temporary storage. When a null
/// pointer is passed the required allocation size in bytes is written to `storage_size` and the
/// function returns `hipSuccess` without performing the merge operation.
/// \param [in,out] storage_size Reference to size in bytes of `temporary_storage`.
/// \param [in,out] data Iterator to the first value to merge.
/// \param [in] left_size Number of elements in the first input range.
/// \param [in] right_size Number of elements in the second input range.
/// \param [in] compare_function Binary operation function that will be used for comparison. The
/// signature of the function should be equivalent to the following: `bool f(const T &a, const T &b);`.
/// The signature does not need to have `const &`, but the function object must not modify
/// the objects passed to it. The default value is `BinaryFunction()`.
/// \param [in] stream The HIP stream object. Default is `0` (`hipDefaultStream`).
/// \param [in] debug_synchronous If `true`, forces a device synchronization after every kernel
/// launch in order to check for errors. Default value is `false`.
///
/// \returns \p hipSuccess `0` after succesful sort; otherwise a HIP runtime error of type
/// `hipError_t`.
///
/// \par Example
/// \parblock
/// \code{.cpp}
/// #include <rocprim/rocprim.hpp>
///
/// size_t left_size;  // e.g. 4
/// size_t right_size; // e.g. 4
/// int*   data;       // e.g. [1, 3, 5, 7, 0, 2, 4, 6]
///
/// size_t temporary_storage_size_bytes;
/// void * temporary_storage_ptr = nullptr;
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
///
/// hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
///
/// rocprim::merge_inplace(
///     temporary_storage_ptr,
///     temporary_storage_size_bytes,
///     data,
///     left_size,
///     right_size);
/// \endcode
/// \endparblock
template<class Config = default_config,
         class Iterator,
         class BinaryFunction
         = ::rocprim::less<typename std::iterator_traits<Iterator>::value_type>>
inline hipError_t merge_inplace(void*             temporary_storage,
                                size_t&           storage_size,
                                Iterator          data,
                                size_t            left_size,
                                size_t            right_size,
                                BinaryFunction    compare_function  = BinaryFunction(),
                                const hipStream_t stream            = 0,
                                bool              debug_synchronous = false)
{
    using config = detail::default_or_custom_config<Config, merge_inplace_config<>>;

    constexpr size_t global_block_size      = config::global_merge_block_size;
    constexpr size_t block_block_size       = config::block_merge_block_size;
    constexpr size_t block_items_per_thread = config::block_merge_items_per_thread;

    const size_t total_size = left_size + right_size;

    using impl = detail::merge_inplace_impl<global_block_size,
                                            block_block_size,
                                            block_items_per_thread,
                                            Iterator,
                                            size_t,
                                            BinaryFunction>;

    typename impl::offset_t* work_storage    = nullptr;
    typename impl::pivot_t*  pivot_storage   = nullptr;
    typename impl::offset_t* scratch_storage = nullptr;

    size_t num_divisions = impl::get_num_global_divisions(left_size, right_size);

    hipError_t result = detail::temp_storage::partition(
        temporary_storage,
        storage_size,
        detail::temp_storage::make_linear_partition(
            detail::temp_storage::ptr_aligned_array(&work_storage, 2ULL << num_divisions),
            detail::temp_storage::ptr_aligned_array(&pivot_storage, 1ULL << num_divisions),
            detail::temp_storage::ptr_aligned_array(&scratch_storage, 2)));

    if(result != hipSuccess || temporary_storage == nullptr)
    {
        if(debug_synchronous)
        {
            std::cout << "device_merge_inplace\n"
                      << "  left  size     : " << left_size << "\n"
                      << "  right size     : " << right_size << "\n"
                      << "  num iterations : " << num_divisions << "\n"
                      << "  requires " << storage_size << " bytes of temporary storage"
                      << std::endl;
        }
        return result;
    }

    if(left_size == 0 || right_size == 0)
        return hipSuccess;

    int max_grid_dim;
    result = detail::grid_dim_for_max_active_blocks(max_grid_dim,
                                                    global_block_size,
                                                    impl::merge_inplace_kernel,
                                                    stream);
    if(result != hipSuccess)
        return result;

    const int global_grid_size = min(static_cast<size_t>(max_grid_dim),
                                     rocprim::detail::ceiling_div(total_size, global_block_size));

    void* kernel_args[] = {
        &work_storage,
        &pivot_storage,
        &scratch_storage,
        &data,
        &left_size,
        &right_size,
        &compare_function,
    };

    std::chrono::high_resolution_clock::time_point t_start, t_stop;

    if(debug_synchronous)
    {
        std::cout << "merge_inplace_kernel\n"
                  << "  grid_size  : " << global_grid_size << "\n"
                  << "  block_size : " << global_block_size << std::endl;
        t_start = std::chrono::high_resolution_clock::now();
    }

    result = hipLaunchCooperativeKernel(impl::merge_inplace_kernel,
                                        global_grid_size,
                                        global_block_size,
                                        kernel_args,
                                        0,
                                        stream);
    if(result != hipSuccess)
        return result;

    if(debug_synchronous)
    {
        result = hipStreamSynchronize(stream);
        if(result != hipSuccess)
            return result;

        t_stop = std::chrono::high_resolution_clock::now();
        std::cout << "  "
                  << std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(t_stop
                                                                                          - t_start)
                         .count()
                  << " ms" << std::endl;
    }

    result = hipGetLastError();
    if(result != hipSuccess)
        return result;

    const size_t block_merge_grid_size = rocprim::detail::ceiling_div(total_size, block_block_size);

    if(debug_synchronous)
    {
        std::cout << "block_merge_kernel\n"
                  << "  grid_size     : " << block_merge_grid_size << "\n"
                  << "  block_size    : " << block_block_size << std::endl;
        t_start = std::chrono::high_resolution_clock::now();
    }

    impl::block_merge_kernel<<<block_merge_grid_size, block_block_size, 0, stream>>>(
        data,
        left_size + right_size,
        compare_function,
        work_storage,
        scratch_storage);

    result = hipGetLastError();
    if(result != hipSuccess)
        return result;

    if(debug_synchronous)
    {
        result = hipStreamSynchronize(stream);
        if(result != hipSuccess)
            return result;

        t_stop = std::chrono::high_resolution_clock::now();
        std::cout << "  "
                  << std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(t_stop
                                                                                          - t_start)
                         .count()
                  << " ms" << std::endl;
    }

    return hipSuccess;
}

/// @}
// end of group devicemodule

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DEVICE_DEVICE_MERGE_INPLACE_HPP_
