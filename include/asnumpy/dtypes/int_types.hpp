/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

// Prevent multiple inclusion
#pragma once

#include <acl/acl.h>
#include <cstdint>
#include <limits>
#include <sstream>
#include <type_traits>

namespace asnumpy {
namespace dtypes {

// Sub-byte integer template base class
// stores N-bit integer in the low N bits of a byte (uint8_t); high bits ignored
template <int N, typename UnderlyingTy> class SubByteInt {
  private:
    UnderlyingTy v_;
    using SignedUnderlyingTy = std::make_signed_t<UnderlyingTy>;
    using UnsignedUnderlyingTy = std::make_unsigned_t<UnderlyingTy>;
    static constexpr int kUnderlyingBits = std::numeric_limits<UnsignedUnderlyingTy>::digits;

    static_assert(std::is_same_v<UnderlyingTy, uint8_t> || std::is_same_v<UnderlyingTy, int8_t>,
                  "underlying type must be signed or unsigned 8-bit integer");

    // mask high bits
    static inline constexpr UnderlyingTy Mask(UnderlyingTy v) {
        return static_cast<UnsignedUnderlyingTy>(static_cast<UnsignedUnderlyingTy>(v) << (kUnderlyingBits - N)) >>
               (kUnderlyingBits - N);
    }

        // mask high bits and sign-extend for signed types
    static inline constexpr UnderlyingTy ExtendToFullWidth(UnderlyingTy v) {
        return static_cast<UnderlyingTy>(static_cast<UnderlyingTy>(v) << (kUnderlyingBits - N)) >>
               (kUnderlyingBits - N);
    }

    // convert to full-width UnderlyingTy value
    inline constexpr UnderlyingTy IntValue() const { return ExtendToFullWidth(v_); }

  public:
    using underlying_type = UnderlyingTy;
    static constexpr int bits = N;
    static constexpr int digits = std::is_signed_v<UnderlyingTy> ? N - 1 : N;

    constexpr SubByteInt() noexcept : v_(0) {}
    constexpr SubByteInt(const SubByteInt& other) noexcept = default;
    constexpr SubByteInt(SubByteInt&& other) noexcept = default;
    constexpr SubByteInt& operator=(const SubByteInt& other) = default;
    constexpr SubByteInt& operator=(SubByteInt&&) = default;

    explicit constexpr SubByteInt(UnderlyingTy val) : v_(Mask(val)) {}

    template <typename T> explicit constexpr SubByteInt(T t) : SubByteInt(static_cast<UnderlyingTy>(t)) {}

    static constexpr SubByteInt highest() { return SubByteInt((1 << digits) - 1); }
    static constexpr SubByteInt lowest() {
        return std::is_signed_v<UnderlyingTy> ? SubByteInt(1) << digits : SubByteInt(0);
    }

    // type conversion
    template <typename T> explicit constexpr operator T() const { return static_cast<T>(IntValue()); }

    // negation
    constexpr SubByteInt operator-() const { return SubByteInt(-v_); }

    // arithmetic operators
    constexpr SubByteInt operator+(const SubByteInt& other) const { return SubByteInt(v_ + other.v_); }
    constexpr SubByteInt operator-(const SubByteInt& other) const { return SubByteInt(v_ - other.v_); }
    constexpr SubByteInt operator*(const SubByteInt& other) const { return SubByteInt(v_ * other.v_); }
    constexpr SubByteInt operator/(const SubByteInt& other) const {
        const UnderlyingTy denom = other.IntValue();
        if (denom == 0) {
            return SubByteInt(0);
        }
        return SubByteInt(IntValue() / denom);
    }
    constexpr SubByteInt operator%(const SubByteInt& other) const {
        const UnderlyingTy denom = other.IntValue();
        if (denom == 0) {
            return SubByteInt(0);
        }
        return SubByteInt((IntValue() % denom));
    }

    // bitwise operators
    constexpr SubByteInt operator&(const SubByteInt& other) const { return SubByteInt(v_ & other.v_); }
    constexpr SubByteInt operator|(const SubByteInt& other) const { return SubByteInt(v_ | other.v_); }
    constexpr SubByteInt operator^(const SubByteInt& other) const { return SubByteInt(v_ ^ other.v_); }
    constexpr SubByteInt operator~() const { return SubByteInt(~v_); }
    constexpr SubByteInt operator>>(int amount) const { return SubByteInt(IntValue() >> amount); }
    constexpr SubByteInt operator<<(int amount) const { return SubByteInt(v_ << amount); }

    // comparison operators
    constexpr bool operator==(const SubByteInt& other) const { return Mask(v_) == Mask(other.v_); }
    constexpr bool operator!=(const SubByteInt& other) const { return Mask(v_) != Mask(other.v_); }
    constexpr bool operator<(const SubByteInt& other) const { return IntValue() < other.IntValue(); }
    constexpr bool operator>(const SubByteInt& other) const { return IntValue() > other.IntValue(); }
    constexpr bool operator<=(const SubByteInt& other) const { return IntValue() <= other.IntValue(); }
    constexpr bool operator>=(const SubByteInt& other) const { return IntValue() >= other.IntValue(); }

    // comparison with int64_t
    constexpr bool operator==(int64_t other) const { return IntValue() == other; }
    constexpr bool operator!=(int64_t other) const { return IntValue() != other; }
    constexpr bool operator<(int64_t other) const { return IntValue() < other; }
    constexpr bool operator>(int64_t other) const { return IntValue() > other; }
    constexpr bool operator<=(int64_t other) const { return IntValue() <= other; }
    constexpr bool operator>=(int64_t other) const { return IntValue() >= other; }

    friend constexpr bool operator==(int64_t a, const SubByteInt& b) { return a == b.IntValue(); }
    friend constexpr bool operator!=(int64_t a, const SubByteInt& b) { return a != b.IntValue(); }
    friend constexpr bool operator<(int64_t a, const SubByteInt& b) { return a < b.IntValue(); }
    friend constexpr bool operator>(int64_t a, const SubByteInt& b) { return a > b.IntValue(); }
    friend constexpr bool operator<=(int64_t a, const SubByteInt& b) { return a <= b.IntValue(); }
    friend constexpr bool operator>=(int64_t a, const SubByteInt& b) { return a >= b.IntValue(); }

    // increment / decrement
    constexpr SubByteInt& operator++() {
        v_ = Mask(v_ + 1);
        return *this;
    }
    constexpr SubByteInt operator++(int) {
        SubByteInt orig = *this;
        this->operator++();
        return orig;
    }
    constexpr SubByteInt& operator--() {
        v_ = Mask(v_ - 1);
        return *this;
    }
    constexpr SubByteInt operator--(int) {
        SubByteInt orig = *this;
        this->operator--();
        return orig;
    }

    // compound assignment operators
    constexpr SubByteInt& operator+=(const SubByteInt& other) {
        *this = *this + other;
        return *this;
    }
    constexpr SubByteInt& operator-=(const SubByteInt& other) {
        *this = *this - other;
        return *this;
    }
    constexpr SubByteInt& operator*=(const SubByteInt& other) {
        *this = *this * other;
        return *this;
    }
    constexpr SubByteInt& operator/=(const SubByteInt& other) {
        const UnderlyingTy denom = other.IntValue();
        if (denom == 0) {
            v_ = 0;
            return *this;
        }
        *this = SubByteInt(IntValue() / denom);
        return *this;
    }
    constexpr SubByteInt& operator%=(const SubByteInt& other) {
        const UnderlyingTy denom = other.IntValue();
        if (denom == 0) {
            v_ = 0;
            return *this;
        }
        *this = SubByteInt(IntValue() % denom);
        return *this;
    }
    constexpr SubByteInt& operator&=(const SubByteInt& other) {
        *this = *this & other;
        return *this;
    }
    constexpr SubByteInt& operator|=(const SubByteInt& other) {
        *this = *this | other;
        return *this;
    }
    constexpr SubByteInt& operator^=(const SubByteInt& other) {
        *this = *this ^ other;
        return *this;
    }
    constexpr SubByteInt& operator>>=(int amount) {
        *this = *this >> amount;
        return *this;
    }
    constexpr SubByteInt& operator<<=(int amount) {
        *this = *this << amount;
        return *this;
    }

    // stream output
    friend std::ostream& operator<<(std::ostream& os, const SubByteInt& num) {
        os << static_cast<int16_t>(num);
        return os;
    }

    // to string
    std::string ToString() const {
        std::ostringstream os;
        os << static_cast<int16_t>(*this);
        return os.str();
    }
};

// int4: 4-bit signed integer, range [-8, 7]
class int4 : public SubByteInt<4, int8_t> {
  private:
    using Base = SubByteInt<4, int8_t>;

  public:
    using Base::Base;

    // get ACL enum
    static constexpr aclDataType getACLenum() { return ACL_INT4; }

    // constants
    static constexpr int kBits = 4;
    static constexpr int kMin = -8;
    static constexpr int kMax = 7;
};

// uint1: 1-bit unsigned integer, range [0, 1]
class uint1 : public SubByteInt<1, uint8_t> {
  private:
    using Base = SubByteInt<1, uint8_t>;

  public:
    using Base::Base;

    // get ACL enum
    static constexpr aclDataType getACLenum() { return ACL_UINT1; }

    // constants
    static constexpr int kBits = 1;
    static constexpr int kMin = 0;
    static constexpr int kMax = 1;
};

} // namespace dtypes
} // namespace asnumpy

// std::numeric_limits specialization
namespace std {

template <typename T, bool IsSigned, bool IsModulo, int Digits> struct asnumpy_subbyte_numeric_limits_common {
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = IsSigned;
    static constexpr bool is_integer = true;
    static constexpr bool is_exact = true;
    static constexpr bool has_infinity = false;
    static constexpr bool has_quiet_NaN = false;
    static constexpr bool has_signaling_NaN = false;
    static constexpr float_denorm_style has_denorm = denorm_absent;
    static constexpr bool has_denorm_loss = false;
    static constexpr float_round_style round_style = round_toward_zero;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_modulo = IsModulo;
    static constexpr int digits = Digits;
    static constexpr int digits10 = 0;
    static constexpr int max_digits10 = 0;
    static constexpr int radix = 2;
    static constexpr int min_exponent = 0;
    static constexpr int min_exponent10 = 0;
    static constexpr int max_exponent = 0;
    static constexpr int max_exponent10 = 0;
    static constexpr bool traps = true;
    static constexpr bool tinyness_before = false;
};

template <>
struct numeric_limits<asnumpy::dtypes::int4>
    : asnumpy_subbyte_numeric_limits_common<asnumpy::dtypes::int4, true, false, 3> {

    static constexpr asnumpy::dtypes::int4 min() noexcept { return asnumpy::dtypes::int4(-8); }
    static constexpr asnumpy::dtypes::int4 lowest() noexcept { return asnumpy::dtypes::int4(-8); }
    static constexpr asnumpy::dtypes::int4 max() noexcept { return asnumpy::dtypes::int4(7); }
    static constexpr asnumpy::dtypes::int4 epsilon() noexcept { return asnumpy::dtypes::int4(0); }
    static constexpr asnumpy::dtypes::int4 round_error() noexcept { return asnumpy::dtypes::int4(0); }
    static constexpr asnumpy::dtypes::int4 infinity() noexcept { return asnumpy::dtypes::int4(0); }
    static constexpr asnumpy::dtypes::int4 quiet_NaN() noexcept { return asnumpy::dtypes::int4(0); }
    static constexpr asnumpy::dtypes::int4 signaling_NaN() noexcept { return asnumpy::dtypes::int4(0); }
    static constexpr asnumpy::dtypes::int4 denorm_min() noexcept { return asnumpy::dtypes::int4(0); }
};

template <>
struct numeric_limits<asnumpy::dtypes::uint1>
    : asnumpy_subbyte_numeric_limits_common<asnumpy::dtypes::uint1, false, true, 1> {

    static constexpr asnumpy::dtypes::uint1 min() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 lowest() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 max() noexcept { return asnumpy::dtypes::uint1(1); }
    static constexpr asnumpy::dtypes::uint1 epsilon() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 round_error() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 infinity() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 quiet_NaN() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 signaling_NaN() noexcept { return asnumpy::dtypes::uint1(0); }
    static constexpr asnumpy::dtypes::uint1 denorm_min() noexcept { return asnumpy::dtypes::uint1(0); }
};

} // namespace std
