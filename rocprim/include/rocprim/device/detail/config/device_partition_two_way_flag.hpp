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

#ifndef ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_TWO_WAY_FLAG_HPP_
#define ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_TWO_WAY_FLAG_HPP_

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
struct default_partition_two_way_flag_config : default_partition_config_base<data_type>::type
{};

template<class data_type>
struct default_partition_two_way_flag_config<static_cast<unsigned int>(target_arch::gfx1030),
                                             data_type>
    : default_partition_config_base<data_type, 15>::type
{};

} // end namespace detail

END_ROCPRIM_NAMESPACE

/// @}
// end of group primitivesmodule_deviceconfigs

#endif // ROCPRIM_DEVICE_DETAIL_CONFIG_DEVICE_PARTITION_TWO_WAY_FLAG_HPP_
