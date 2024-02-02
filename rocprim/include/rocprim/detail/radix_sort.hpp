// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_DETAIL_RADIX_SORT_HPP_
#define ROCPRIM_DETAIL_RADIX_SORT_HPP_

#include <initializer_list>
#include <type_traits>
#include <utility>

#include "../config.hpp"
#include "../functional.hpp"
#include "../type_traits.hpp"
#include "../types/tuple.hpp"
#include "various.hpp"

BEGIN_ROCPRIM_NAMESPACE
namespace detail
{

// Encode and decode integral and floating point values for radix sort in such a way that preserves
// correct order of negative and positive keys (i.e. negative keys go before positive ones,
// which is not true for a simple reinterpetation of the key's bits).

// Digit extractor takes into account that (+0.0 == -0.0) is true for floats,
// so both +0.0 and -0.0 are reflected into the same bit pattern for digit extraction.
// Maximum digit length is 32.

template<class Key, class BitKey, class Enable = void>
struct radix_key_codec_integral { };

template<class Key, class BitKey>
struct radix_key_codec_integral<Key, BitKey, typename std::enable_if<::rocprim::is_unsigned<Key>::value>::type>
{
    using bit_key_type = BitKey;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bit_key_type encode(Key key)
    {
        return ::rocprim::detail::bit_cast<bit_key_type>(key);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static Key decode(bit_key_type bit_key)
    {
        return ::rocprim::detail::bit_cast<Key>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<class Key, class BitKey>
struct radix_key_codec_integral<
    Key,
    BitKey,
    typename std::enable_if<std::is_same<Key, __uint128_t>::value>::type>
{
    using bit_key_type = BitKey;

    ROCPRIM_DEVICE ROCPRIM_INLINE static bit_key_type encode(Key key)
    {
        return ::rocprim::detail::bit_cast<bit_key_type>(key);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static Key decode(bit_key_type bit_key)
    {
        return ::rocprim::detail::bit_cast<Key>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<class Key, class BitKey>
struct radix_key_codec_integral<Key, BitKey, typename std::enable_if<::rocprim::is_signed<Key>::value>::type>
{
    using bit_key_type = BitKey;

    static constexpr bit_key_type sign_bit = bit_key_type(1) << (sizeof(bit_key_type) * 8 - 1);

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bit_key_type encode(Key key)
    {
        const auto bit_key = ::rocprim::detail::bit_cast<bit_key_type>(key);
        return sign_bit ^ bit_key;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static Key decode(bit_key_type bit_key)
    {
        bit_key ^= sign_bit;
        return ::rocprim::detail::bit_cast<Key>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<class Key, class BitKey>
struct radix_key_codec_integral<Key,
                                BitKey,
                                typename std::enable_if<std::is_same<Key, __int128_t>::value>::type>
{
    using bit_key_type = BitKey;

    static constexpr bit_key_type sign_bit = bit_key_type(1) << (sizeof(bit_key_type) * 8 - 1);

    ROCPRIM_DEVICE ROCPRIM_INLINE static bit_key_type encode(Key key)
    {
        const auto bit_key = ::rocprim::detail::bit_cast<bit_key_type>(key);
        return sign_bit ^ bit_key;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static Key decode(bit_key_type bit_key)
    {
        bit_key ^= sign_bit;
        return ::rocprim::detail::bit_cast<Key>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<class Key>
struct float_bit_mask;

template<>
struct float_bit_mask<float>
{
    static constexpr uint32_t sign_bit = 0x80000000;
    static constexpr uint32_t exponent = 0x7F800000;
    static constexpr uint32_t mantissa = 0x007FFFFF;
    using bit_type = uint32_t;
};

template<>
struct float_bit_mask<double>
{
    static constexpr uint64_t sign_bit = 0x8000000000000000;
    static constexpr uint64_t exponent = 0x7FF0000000000000;
    static constexpr uint64_t mantissa = 0x000FFFFFFFFFFFFF;
    using bit_type = uint64_t;
};

template<>
struct float_bit_mask<rocprim::bfloat16>
{
    static constexpr uint16_t sign_bit = 0x8000;
    static constexpr uint16_t exponent = 0x7F80;
    static constexpr uint16_t mantissa = 0x007F;
    using bit_type = uint16_t;
};

template<>
struct float_bit_mask<rocprim::half>
{
    static constexpr uint16_t sign_bit = 0x8000;
    static constexpr uint16_t exponent = 0x7C00;
    static constexpr uint16_t mantissa = 0x03FF;
    using bit_type = uint16_t;
};

template<class Key, class BitKey>
struct radix_key_codec_floating
{
    using bit_key_type = BitKey;

    static constexpr bit_key_type sign_bit = float_bit_mask<Key>::sign_bit;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bit_key_type encode(Key key)
    {
        bit_key_type bit_key = ::rocprim::detail::bit_cast<bit_key_type>(key);
        bit_key ^= (sign_bit & bit_key) == 0 ? sign_bit : bit_key_type(-1);
        return bit_key;
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static Key decode(bit_key_type bit_key)
    {
        bit_key ^= (sign_bit & bit_key) == 0 ? bit_key_type(-1) : sign_bit;
        return ::rocprim::detail::bit_cast<Key>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;

        // radix_key_codec_floating::encode() maps 0.0 to 0x8000'0000,
        // and -0.0 to 0x7FFF'FFFF.
        // radix_key_codec::encode() then flips the bits if descending, yielding:
        // value | descending  | ascending   |
        // ----- | ----------- | ----------- |
        //   0.0 | 0x7FFF'FFFF | 0x8000'0000 |
        //  -0.0 | 0x8000'0000 | 0x7FFF'FFFF |
        //
        // For ascending sort, both should be mapped to 0x8000'0000,
        // and for descending sort, both should be mapped to 0x7FFF'FFFF.
        if ROCPRIM_IF_CONSTEXPR(Descending)
        {
            bit_key = bit_key == sign_bit ? static_cast<bit_key_type>(~sign_bit) : bit_key;
        }
        else
        {
            bit_key = bit_key == static_cast<bit_key_type>(~sign_bit) ? sign_bit : bit_key;
        }
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<class Key, class Enable = void>
struct radix_key_codec_base
{
    static_assert(sizeof(Key) == 0,
        "Only integral and floating point types supported as radix sort keys");
};

template<class Key>
struct radix_key_codec_base<
    Key,
    typename std::enable_if<::rocprim::is_integral<Key>::value>::type
> : radix_key_codec_integral<Key, typename std::make_unsigned<Key>::type> { };

template<class Key>
struct radix_key_codec_base<Key,
                            typename std::enable_if<std::is_same<Key, __int128_t>::value>::type>
    : radix_key_codec_integral<Key, __uint128_t>
{};

template<class Key>
struct radix_key_codec_base<Key,
                            typename std::enable_if<std::is_same<Key, __uint128_t>::value>::type>
    : radix_key_codec_integral<Key, __uint128_t>
{};

template<>
struct radix_key_codec_base<bool>
{
    using bit_key_type = unsigned char;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bit_key_type encode(bool key)
    {
        return static_cast<bit_key_type>(key);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bool decode(bit_key_type bit_key)
    {
        return static_cast<bool>(bit_key);
    }

    template<bool Descending>
    ROCPRIM_DEVICE static unsigned int
        extract_digit(bit_key_type bit_key, unsigned int start, unsigned int length)
    {
        unsigned int mask = (1u << length) - 1;
        return static_cast<unsigned int>(bit_key >> start) & mask;
    }
};

template<>
struct radix_key_codec_base<::rocprim::half> : radix_key_codec_floating<::rocprim::half, unsigned short> { };

template<>
struct radix_key_codec_base<::rocprim::bfloat16> : radix_key_codec_floating<::rocprim::bfloat16, unsigned short> { };

template<>
struct radix_key_codec_base<float> : radix_key_codec_floating<float, unsigned int> { };

template<>
struct radix_key_codec_base<double> : radix_key_codec_floating<double, unsigned long long> { };

template<class Key, bool Descending = false>
class radix_key_codec : protected radix_key_codec_base<Key>
{
    using base_type = radix_key_codec_base<Key>;

public:
    using bit_key_type = typename base_type::bit_key_type;

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static bit_key_type encode(Key key)
    {
        bit_key_type bit_key = base_type::encode(key);
        return (Descending ? ~bit_key : bit_key);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static void encode_inplace(Key& key)
    {
        key = ::rocprim::detail::bit_cast<Key>(encode(key));
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static Key decode(bit_key_type bit_key)
    {
        bit_key = (Descending ? ~bit_key : bit_key);
        return base_type::decode(bit_key);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static void decode_inplace(Key& key)
    {
        key = decode(::rocprim::detail::bit_cast<bit_key_type>(key));
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    static unsigned int extract_digit(bit_key_type bit_key, unsigned int start, unsigned int radix_bits)
    {
        return base_type::template extract_digit<Descending>(bit_key, start, radix_bits);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE static unsigned int
        extract_digit_from_key(Key key, unsigned int start, unsigned int radix_bits)
    {
        const auto bit_key = ::rocprim::detail::bit_cast<bit_key_type>(key);
        return extract_digit(bit_key, start, radix_bits);
    }
};

template<bool Descending, class... Args>
class radix_key_codec<::rocprim::tuple<Args...>, Descending>
{
    // TODO: assert all tuple fields are references

private:
    using tuple_t                      = ::rocprim::tuple<Args...>;
    static constexpr size_t tuple_size = sizeof...(Args);

public:
    ROCPRIM_HOST_DEVICE static void encode_inplace(tuple_t& key_tuple)
    {
        encode_inplace_impl(key_tuple, std::index_sequence_for<Args...>());
    }

    ROCPRIM_HOST_DEVICE static void decode_inplace(tuple_t& key_tuple)
    {
        decode_inplace_impl(key_tuple, std::index_sequence_for<Args...>());
    }

    ROCPRIM_HOST_DEVICE static unsigned int extract_digit_from_key(const tuple_t&     key_tuple,
                                                                   const unsigned int start,
                                                                   const unsigned int radix_bits)
    {
        return extract_digit_from_key_impl<tuple_size - 1>(0u,
                                                           key_tuple,
                                                           static_cast<int>(start),
                                                           static_cast<int>(start + radix_bits),
                                                           0);
    }

private:
    template<size_t... Indices>
    ROCPRIM_HOST_DEVICE static void encode_inplace_impl(tuple_t& key_tuple,
                                                        std::index_sequence<Indices...>)
    {
        (void)std::initializer_list<int>{(
            [&]
            {
                using T            = std::decay_t<::rocprim::tuple_element_t<Indices, tuple_t>>;
                using ElementCodec = radix_key_codec<T, Descending>;
                ElementCodec::encode_inplace(::rocprim::get<Indices>(key_tuple));
            }(),
            0)...};
    }

    template<size_t... Indices>
    ROCPRIM_HOST_DEVICE static void decode_inplace_impl(tuple_t& key_tuple,
                                                        std::index_sequence<Indices...>)
    {
        (void)std::initializer_list<int>{(
            [&]
            {
                using T            = std::decay_t<::rocprim::tuple_element_t<Indices, tuple_t>>;
                using ElementCodec = radix_key_codec<T, Descending>;
                ElementCodec::decode_inplace(::rocprim::get<Indices>(key_tuple));
            }(),
            0)...};
    }

    template<int ElementIndex>
    ROCPRIM_HOST_DEVICE static auto extract_digit_from_key_impl(unsigned int digit,
                                                                const tuple_t& /*key_tuple*/,
                                                                const int /*start*/,
                                                                const int /*end*/,
                                                                const int /*previous_bits*/)
        -> std::enable_if_t<(ElementIndex < 0), unsigned int>
    {
        return digit;
    }

    template<int ElementIndex>
    ROCPRIM_HOST_DEVICE static auto extract_digit_from_key_impl(unsigned int   digit,
                                                                const tuple_t& key_tuple,
                                                                const int      start,
                                                                const int      end,
                                                                const int      previous_bits)
        -> std::enable_if_t<(ElementIndex >= 0), unsigned int>
    {
        using T            = std::decay_t<::rocprim::tuple_element_t<ElementIndex, tuple_t>>;
        using bit_key_type = typename radix_key_codec<T, Descending>::bit_key_type;
        constexpr int current_element_bits = 8 * sizeof(T);

        const int total_extracted_bits    = end - start;
        const int current_element_end_bit = previous_bits + current_element_bits;
        if(start < current_element_end_bit && end > previous_bits)
        {
            // unsigned integral representation of the current tuple element
            const auto element_value = ::rocprim::detail::bit_cast<bit_key_type>(
                ::rocprim::get<ElementIndex>(key_tuple));

            const int bits_extracted_previously = ::rocprim::max(0, previous_bits - start);

            // start of the bit range copied from the current tuple element
            const int current_start_bit = ::rocprim::max(0, start - previous_bits);

            // end of the bit range copied from the current tuple element
            const int current_end_bit = ::rocprim::min(current_element_bits,
                                                       current_start_bit + total_extracted_bits
                                                           - bits_extracted_previously);

            // number of bits extracted from the current tuple element
            const int current_length = current_end_bit - current_start_bit;

            // bits extracted from the current tuple element, aligned to LSB
            const unsigned int current_extract
                = (element_value >> current_start_bit) & ((1u << current_length) - 1);

            digit |= current_extract << bits_extracted_previously;
        }
        return extract_digit_from_key_impl<ElementIndex - 1>(digit,
                                                             key_tuple,
                                                             start,
                                                             end,
                                                             previous_bits + current_element_bits);
    }
};

} // end namespace detail
END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_DETAIL_RADIX_SORT_HPP_
