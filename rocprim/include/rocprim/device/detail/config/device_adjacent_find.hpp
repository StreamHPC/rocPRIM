// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_FIND_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_FIND_HPP_

#include "../../../type_traits.hpp"
#include "../device_config_helper.hpp"

#include <type_traits>

/* DO NOT EDIT THIS FILE
 * This file is automatically generated by `/scripts/autotune/create_optimization.py`.
 * so most likely you want to edit rocprim/device/device_(algo)_config.hpp
 */

/// \addtogroup primitivesmodule_deviceconfigs
/// @{

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{

template<unsigned int arch, class input_type, class enable = void>
struct default_adjacent_find_config : default_adjacent_find_config_base<input_type>::type
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 8>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<512, 4>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<256, 16>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<512, 2>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<256, 8>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<128, 64>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<128, 32>
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<512, 2>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<128, 4>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<512, 8>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<256, 4>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<128, 8>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<64, 4>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx1100),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<128, 32>
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 16>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<128, 16>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<64, 4>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 4>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<128, 16>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<128, 2>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx906),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<128, 2>
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 8>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<512, 4>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<256, 16>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<512, 2>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<512, 4>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<256, 16>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx908),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<64, 16>
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 8>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<512, 4>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<256, 16>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<512, 2>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<64, 64>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<256, 16>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::unknown),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<64, 16>
{};

// Based on input_type = double
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<128, 8>
{};

// Based on input_type = float
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<512, 4>
{};

// Based on input_type = rocprim::half
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2))>> : adjacent_find_config<256, 16>
{};

// Based on input_type = int64_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 8) && (sizeof(input_type) > 4))>>
    : adjacent_find_config<512, 2>
{};

// Based on input_type = int
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 4) && (sizeof(input_type) > 2))>>
    : adjacent_find_config<64, 64>
{};

// Based on input_type = short
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 2) && (sizeof(input_type) > 1))>>
    : adjacent_find_config<256, 16>
{};

// Based on input_type = int8_t
template<class input_type>
struct default_adjacent_find_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    input_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<input_type>::value)
                      && (sizeof(input_type) <= 1))>> : adjacent_find_config<64, 16>
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_ADJACENT_FIND_HPP_