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

#ifndef ROCPRIM_TYPE_TRAITS_INTERFACE_HPP_
#define ROCPRIM_TYPE_TRAITS_INTERFACE_HPP_

#include "type_traits.hpp"

// common macros

// A reverse version of static_assert aims to increase code readability
#ifndef ROCPRIM_DO_NOT_COMPILE_IF
    #define ROCPRIM_DO_NOT_COMPILE_IF(condition, msg) static_assert(!(condition), msg)
#endif

// Wrapper macro for std::enable_if aims to increase code readability
#ifndef ROCPRIM_REQUIRES
    #define ROCPRIM_REQUIRES(...) typename std::enable_if<(__VA_ARGS__)>::type* = nullptr
#endif

// Since every definable traits need to use `is_defined`, this macro reduce the amount of code
#define ROCPRIM_TRAITS_GENERATE_IS_DEFINE(traits_name)                                 \
    template<class InputType, class = void>                                            \
    static constexpr bool is_defined = false;                                          \
    template<class InputType>                                                          \
    static constexpr bool                                                              \
        is_defined<InputType, detail::void_t<typename define<InputType>::traits_name>> \
        = true

BEGIN_ROCPRIM_NAMESPACE

namespace traits
{
// ***Interface for third party types / user types***

/// \brief Users can use this template struct to define tratis for types
template<class T>
struct define
{};

/// ***Definable traits***

/// \brief Trait [is_arithmetic] should be definable because there might be types like `thirdparty::float256`
/// If trait [is_arithmetic] is not defined, then the get return default values
struct is_arithmetic
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_arithmetic);
    template<bool Val>
    struct values
    {
        static constexpr auto is_arithmetic = Val;
    };

    /// \brief For c++ arithmetic types, return true, but will throw compile error when user try to define this trait for them
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>,
                                  "Do not define trait [is_arithmetic] for c++ arithmetic types");
        return values<true>{};
    }

    /// \brief For third party types, if trait [is_arithmetic] not defined, will return default value `false`
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value && !is_defined<InputType>)>
    static constexpr auto get()
    {
        return values<false>{};
    }

    /// \brief For third party types, if trait [is_arithmetic] is defined, then should return its value
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value && is_defined<InputType>)>
    static constexpr auto get()
    {
        return typename define<InputType>::is_arithmetic{};
    }
};

/// \brief for arithmetic, pointers, member_pointers, and null_pointers, they should be scalar types
struct is_scalar
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_scalar);
    template<bool Val>
    struct values
    {
        static constexpr auto is_scalar = Val;
    };

    /// \brief For c++ scalar types, return true, but will throw compile error when user try to define this trait for them
    template<class InputType, ROCPRIM_REQUIRES(std::is_scalar<InputType>::value)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>,
                                  "Do not define trait [is_scalar] for c++ scalar types");
        return values<true>{};
    }

    /// \brief For third party types, if trait [is_scalar] not defined, will return default value `false`
    /// For rocprim rocprim or third party types that defined trait [is_arithmetic] as true the result should be `true`
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_scalar<InputType>::value && !is_defined<InputType>)>
    static constexpr auto get()
    {
        return values<is_arithmetic::get<InputType>().is_arithmetic>{};
    }

    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_scalar<InputType>::value && is_defined<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_arithmetic::get<InputType>().is_arithmetic !=
                typename define<InputType>::is_scalar{}.is_scalar,
            "Trait [is_arithmetic] and trait [is_scalar] should have the same value");
        return typename define<InputType>::is_scalar{};
    }
};

struct is_float_or_int
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_float_or_int);

    template<int Val>
    struct values
    {
        static constexpr auto is_float_or_int = Val;
    };

    // c++ arithmetic types
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    { // cpp arithmetic types are either floating point or integral
        return values < std::is_floating_point<InputType>::value ? 0 : 1 > {};
    }

    // rocprim arithmetic types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && is_arithmetic::get<InputType>().is_arithmetic)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(!is_defined<InputType>,
                                  "You must define trait [is_float_or_int] for arithmetic types");
        return typename define<InputType>::is_float_or_int{};
    }

    // other types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && !is_arithmetic::get<InputType>().is_arithmetic)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait [is_float_or_int] for non-arithmetic types");
        return values<2>{};
    }
};

struct is_signed_or_unsigned
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_signed_or_unsigned);

    template<int Val>
    struct values
    {
        static constexpr auto is_signed_or_unsigned = Val;
    };

    // c++ arithmetic types
    template<class InputType, ROCPRIM_REQUIRES(std::is_arithmetic<InputType>::value)>
    static constexpr auto get()
    { // cpp arithmetic types are either signed point or unsignned
        return values < std::is_signed<InputType>::value ? 0 : 1 > {};
    }

    // rocprim arithmetic integral
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && is_arithmetic::get<InputType>().is_arithmetic
                              && is_float_or_int::get<InputType>().is_float_or_int == 1)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(!is_defined<InputType>,
                                  "Trait [is_signed_or_unsigned] is required for arithmetic "
                                  "integral types, please define");
        return typename define<InputType>::is_signed_or_unsigned{};
    }

    // rocprim arithmetic non-integral
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && is_arithmetic::get<InputType>().is_arithmetic
                              && is_float_or_int::get<InputType>().is_float_or_int != 1)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait [is_signed_or_unsigned] for arithmetic non-integral types");
        return values<2>{};
    }

    // other types
    template<class InputType,
             ROCPRIM_REQUIRES(!std::is_arithmetic<InputType>::value
                              && !is_arithmetic::get<InputType>().is_arithmetic)>
    static constexpr auto get()
    { //  for other types, trait is_floating_point is a must
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_defined<InputType>,
            "You cannot define trait [is_signed_or_unsigned] for non-arithmetic types");
        return values<2>{};
    }
};

/// \brief `float_bit_mask` is a trait that `Must be defined` for `floating_point` types,
/// but `Undefinable` for other types
struct float_bit_mask
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(float_bit_mask);

    template<class InputType, class = void>
    static constexpr auto has_old_float_bit_mask = false;
    template<class InputType>
    static constexpr auto has_old_float_bit_mask<
        InputType,
        detail::void_t<decltype(rocprim::detail::float_bit_mask<InputType>{})>>
        = true;

    template<class BitType, BitType SignBit, BitType Exponent, BitType Mantissa>
    struct values
    {
        static constexpr BitType sign_bit = SignBit;
        static constexpr BitType exponent = Exponent;
        static constexpr BitType mantissa = Mantissa;
    };

    // if defined traits using new interface
    template<class InputType, ROCPRIM_REQUIRES(is_defined<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_float_or_int::get<InputType>().is_float_or_int != 0,
            "You cannot use trait [float_bit_mask] for [non-floating_point] types");
        return typename define<InputType>::float_bit_mask{};
    }

    /// This function acts as a bridge for old interface
    /// Will be removed in certain version
    template<class InputType,
             ROCPRIM_REQUIRES(!is_defined<InputType> && has_old_float_bit_mask<InputType>)>
    static constexpr auto get()
    {
        using mask = typename ::rocprim::detail::float_bit_mask<InputType>;
        return values<typename mask::bit_type, mask::sign_bit, mask::exponent, mask::mantissa>{};
    }

    template<class InputType,
             ROCPRIM_REQUIRES(!is_defined<InputType> && !has_old_float_bit_mask<InputType>)>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(
            is_float_or_int::get<InputType>().is_float_or_int != 0,
            "You cannot use trait [float_bit_mask] for [non-floating_point] types");
        ROCPRIM_DO_NOT_COMPILE_IF(is_float_or_int::get<InputType>().is_float_or_int == 0,
                                  "Trait [float_bit_mask] is required for [floating_point] types");
        return values<int, 0, 0, 0>{};
    }
};

/// ***Undefinable traits***
/// If user defines these traits, compiler will rise AN error

/// \brief Trait [is_fundamental] should not be definable, because it is the union of `std::is_fundamental` and `rocprim::traits::is_arithmetic`.
struct is_fundamental
{
    ROCPRIM_TRAITS_GENERATE_IS_DEFINE(is_fundamental);
    template<bool Val>
    struct values
    {
        static constexpr auto is_fundamental = Val;
    };

    template<class InputType>
    static constexpr auto get()
    {
        ROCPRIM_DO_NOT_COMPILE_IF(is_defined<InputType>, "Trait [is_fundamental] is undefinable");
        return values < std::is_fundamental<InputType>::value
               || is_arithmetic::get<InputType>().is_arithmetic > {};
    }
};

// ***Interface for rocprim algorithms***

/// \brief This struct is designed for rocprim algorithms to get the traits information form C++ arithematic types,
/// rocprim types and third party types.
///
///  - All member functions are `Compiled only when called`.
///
///  - For different algorithms, different traits are required.
///
///  - Operators of third party types should be inplemented if they want to use rocprim algorithms. Now they can implement some type traits.
///
///  - Implementing type traits is similar to implementing operators -- We only and must implement those needed by target algorithm.
template<class T>
struct get
{
    constexpr auto is_arithmetic() const
    {
        return rocprim::traits::is_arithmetic{}.get<T>().is_arithmetic;
    };

    constexpr auto is_fundamental() const
    {
        return rocprim::traits::is_fundamental{}.get<T>().is_fundamental;
    };

    constexpr auto is_compound() const
    {
        return !rocprim::traits::is_fundamental{}.get<T>().is_fundamental;
    }

    constexpr auto is_floating_point() const
    {
        return rocprim::traits::is_float_or_int{}.get<T>().is_float_or_int == 0;
    };

    constexpr auto is_integral() const
    {
        return rocprim::traits::is_float_or_int{}.get<T>().is_float_or_int == 1;
    }

    constexpr auto is_signed() const
    {
        return rocprim::traits::is_signed_or_unsigned{}.get<T>().is_signed_or_unsigned == 0;
    }

    constexpr auto is_unsigned() const
    {
        return rocprim::traits::is_signed_or_unsigned{}.get<T>().is_signed_or_unsigned == 1;
    }

    constexpr auto is_scalar() const
    {
        return rocprim::traits::is_scalar{}.get<T>().is_scalar;
    }

    constexpr auto float_bit_mask() const
    {
        return rocprim::traits::float_bit_mask{}.get<T>();
    };
};

} // namespace traits

// Traits definition for rocprim arithmetic types and additional traits for c++ arithmetic types

// C++ types

template<>
struct traits::define<float>
{
    using float_bit_mask
        = traits::float_bit_mask::values<uint32_t, 0x80000000, 0x7F800000, 0x007FFFFF>;
};

template<>
struct traits::define<double>
{
    using float_bit_mask = traits::float_bit_mask::
        values<uint64_t, 0x8000000000000000, 0x7FF0000000000000, 0x000FFFFFFFFFFFFF>;
};

// rocprim arithmetic types

template<>
struct traits::define<rocprim::bfloat16>
{
    using is_arithmetic   = traits::is_arithmetic::values<true>;
    using is_float_or_int = traits::is_float_or_int::values<0>;
    using float_bit_mask  = traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};

template<>
struct traits::define<rocprim::half>
{
    using is_arithmetic   = traits::is_arithmetic::values<true>;
    using is_float_or_int = traits::is_float_or_int::values<0>;
    using float_bit_mask  = traits::float_bit_mask::values<uint16_t, 0x8000, 0x7F80, 0x007F>;
};

template<>
struct traits::define<rocprim::int128_t>
{
    using is_arithmetic         = traits::is_arithmetic::values<true>;
    using is_float_or_int       = traits::is_float_or_int::values<1>;
    using is_signed_or_unsigned = traits::is_signed_or_unsigned::values<0>;
};

template<>
struct traits::define<rocprim::uint128_t>
{
    using is_arithmetic         = traits::is_arithmetic::values<true>;
    using is_float_or_int       = traits::is_float_or_int::values<1>;
    using is_signed_or_unsigned = traits::is_signed_or_unsigned::values<1>;
};

END_ROCPRIM_NAMESPACE

#endif
