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

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_FLAG_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_FLAG_HPP_

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

template<unsigned int arch, class data_type, class enable = void>
struct default_partition_flag_config : default_partition_config_base<data_type>::type
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<512, 4>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<512, 4>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<512, 12>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<512, 4>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<512, 4>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<512, 12>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1030),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<128, 28>
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<512, 7>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<256, 15>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<128, 19>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<256, 7>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<256, 12>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<128, 19>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx1102),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<128, 20>
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<256, 7>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<256, 13>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<256, 28>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<256, 7>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<256, 13>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<256, 28>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx906),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<256, 20>
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<128, 7>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<128, 12>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<256, 28>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<128, 7>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<128, 12>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<256, 28>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx908),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<256, 28>
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<128, 7>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<128, 12>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<256, 28>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<128, 7>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<128, 12>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<256, 28>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::unknown),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<256, 28>
{};

// Based on data_type = double
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 8)
                      && (sizeof(data_type) > 4))>> : select_config<128, 7>
{};

// Based on data_type = float
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value) && (sizeof(data_type) <= 4)
                      && (sizeof(data_type) > 2))>> : select_config<128, 12>
{};

// Based on data_type = rocprim::half
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2))>> : select_config<256, 28>
{};

// Based on data_type = int64_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 8) && (sizeof(data_type) > 4))>>
    : select_config<128, 7>
{};

// Based on data_type = int
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 4) && (sizeof(data_type) > 2))>>
    : select_config<128, 12>
{};

// Based on data_type = short
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 2) && (sizeof(data_type) > 1))>>
    : select_config<256, 28>
{};

// Based on data_type = int8_t
template<class data_type>
struct default_partition_flag_config<
    static_cast<unsigned int>(target_arch::gfx90a),
    data_type,
    std::enable_if_t<(!bool(rocprim::is_floating_point<data_type>::value)
                      && (sizeof(data_type) <= 1))>> : select_config<256, 28>
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_FLAG_HPP_
