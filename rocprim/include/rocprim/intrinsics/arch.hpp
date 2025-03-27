// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_INTRINSICS_ARCH_HPP_
#define ROCPRIM_INTRINSICS_ARCH_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \brief Utilities to query architecture details.
namespace arch
{

/// \brief Utilities to query wavefront details.
namespace wavefront
{

/// \brief Return the number of threads in the wavefront.
///
/// This function is not `constexpr`.
    ROCPRIM_DEVICE ROCPRIM_INLINE
unsigned int size()
{
    // This function is **not** constexpr because it will
    // be using '__builtin_amdgcn_wavefrontsize()'.
    return warpSize;
}

/// \brief Return the minimum number of threads in the wavefront.
///
/// This function can be used to setup compile time allocation of
/// global or shared memory.
///
/// \par Example
/// \parblock
/// The example below shows how shared memory can be allocated
/// to collect per warp results.
/// \code{.cpp}
/// constexpr auto total_items = 1024;
/// constexpr auto max_warps   = total_items / arch::min_size();
///
/// // If we want to use shared memory to exchange data
/// // between warps, we can allocate it as:
/// __shared int per_warp_results[max_warps];
/// \endcode
/// \endparblock
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
constexpr unsigned int min_size()
{
#if __HIP_DEVICE_COMPILE__
    #if ROCPRIM_NAVI
    return 32u;
    #endif
    return 64u;
#else
    return ROCPRIM_WARP_SIZE_32;
#endif
}

/// \brief Return the maximum number of threads in the wavefront.
///
/// This function can be used to setup compile time allocation of
/// global or shared memory.
///
/// \par Example
/// \parblock
/// The example below shows how an array can be allocated
/// to collect a single warp's results.
/// \code{.cpp}
/// constexpr auto items_per_thread = 2;
///
/// // If we want to collect all the elements in a single array
/// // on a single thread, we can allocate it as:
/// int single_warp[items_per_thread * arch::max_size()];
/// \endcode
/// \endparblock
    ROCPRIM_HOST_DEVICE ROCPRIM_INLINE
constexpr unsigned int max_size()
{
#if __HIP_DEVICE_COMPILE__
    return min_size();
#else
    return ROCPRIM_WARP_SIZE_64;
#endif
}

enum class target_t
{
    size32,
    size64,
    dynamic,
};

/// \brief Returns the hardware wavefront size of the current
/// compile target.
///
/// On host this will return \p target_t::dynamic . On device
/// this return \p target_t::size32 , \p target_t::size64 , or
/// when targeting SPIR-V \p target::dynamic .
constexpr target_t target() noexcept
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__SPIRV__)
    // SPIR-V and host compile both have unknown compile size.
    return target_t::dynamic;
#else
    // The wavefront size is exactly known.
    static_assert(min_size() == max_size());

    if constexpr(min_size() == ROCPRIM_WARP_SIZE_32)
    {
        return target_t::size32;
    }
    return target_t::size64;
#endif
}

/// \brief Returns the numerical wavefront size from a
/// given \p target_t .
///
/// This function has no implementation for
/// \p target_t::dynamic .
template<target_t target>
constexpr unsigned int size_from_target() = delete;
template<>
constexpr unsigned int size_from_target<target_t::size32>()
{
    return ROCPRIM_WARP_SIZE_32;
}
template<>
constexpr unsigned int size_from_target<target_t::size64>()
{
    return ROCPRIM_WARP_SIZE_64;
}

}; // namespace wavefront

} // namespace arch

namespace detail
{

template<typename Impl32, typename Impl64>
struct dispatch_wave_size
{
    union storage_type
    {
        typename Impl32::storage_type wave32;
        typename Impl64::storage_type wave64;
    };

    template<typename F, typename... Args>
    ROCPRIM_HOST_DEVICE
    auto operator()(F exec, Args&&... args)
    {
        // Select either the wave32 or wave64 implementation.
        auto select = [&](auto impl) -> decltype(auto)
        {
            // Given an implementation, execute our callback and
            // pass our re-mapped arguments. The re-mapping selects
            // the appropiate backing 'storage_type' for the chosen
            // implementation.
            return exec(
                // Pass the selected implementation to the callback.
                impl,
                // Map over every argument in the varadic packing...
                [](auto&& arg) -> decltype(auto)
                {
                    // If the argument is 'storage_type'...
                    if constexpr(std::is_same_v<decltype(arg), storage_type&>)
                    { // And we have a wave32 implementation...
                        if constexpr(std::is_same_v<decltype(impl), Impl32>)
                        { // We return the wave32 backing storage!
                            return std::forward<decltype(arg.wave32)&>(arg.wave32);
                        }
                        else
                        { // And otherwise the wave64 backing storage!
                            return std::forward<decltype(arg.wave64)&>(arg.wave64);
                        }
                    }
                    else
                    { // Otherwise, pass argument transparently.
                        return std::forward<decltype(arg)>(arg);
                    }
                }(args)...);
        };

        // Now do the actual implementation selection. The compiler
        // *should* optimize this after lowering, but the extra
        // allocated shared memory due to union is unrecoverable.
        return ::rocprim::arch::wavefront::size() == ROCPRIM_WARP_SIZE_64 ? select(Impl64{})
                                                                          : select(Impl32{});
    }
};

/// \brief Utility function to assert the wavefront size.
///
/// Assertion is done either at run time if we are curenntly
/// compiling for SPIR-V, or if the target is dynamic.
/// Otherwise, we use a static assert.
template<::rocprim::arch::wavefront::target_t target>
struct check_wave_size
{
    /// \brief The assertion to do.
        template<typename P>
    ROCPRIM_FORCE_INLINE ROCPRIM_HOST_DEVICE
        constexpr void operator()(P predicate) const
        {
#if !defined(__HIP_DEVICE_COMPILE__) || ROCPRIM_TARGET_SPIRV
            // When a dynamic wavefront size specializes, we actually
            // don't know if the type is valid or not.
            assert(predicate(::rocprim::arch::wavefront::size()));
#else
            // If we are on device, we do want to statically assert, if possible!
            static_assert(predicate(::rocprim::arch::wavefront::size_from_target<target>()));
#endif
        }
};

template<>
struct check_wave_size<::rocprim::arch::wavefront::target_t::dynamic>
{
        template<typename P>
    ROCPRIM_FORCE_INLINE ROCPRIM_HOST_DEVICE
        void operator()(P predicate) const
        {
            // Since we don't know the wavefront size, we have to
            // do a runtime query.
            assert(predicate(::rocprim::arch::wavefront::size()));
        }
};

} // namespace detail

END_ROCPRIM_NAMESPACE

#endif
