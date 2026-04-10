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

#pragma once

#include <asnumpy/dtypes/float/bit_cast.hpp>
#include <asnumpy/dtypes/float/f32_helpers.hpp>
#include <acl/acl.h>

namespace asnumpy {
namespace dtypes {

class float6_e2m3fn {
private:
    uint8_t rep_;
    struct ConstructFromRepTag {};
    constexpr float6_e2m3fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}

    static uint8_t encode_from_float(float f) {
        uint32_t u = bit_cast<uint32_t>(f);
        uint32_t sign = u >> 31;
        uint32_t exp = (u >> 23) & 0xFFu;
        uint32_t frac = u & 0x7FFFFFu;

        // MX 格式：无 Inf/NaN，有限值范围
        if (exp == 0xFFu) {
            // NaN/Inf -> 最大有限值
            return static_cast<uint8_t>((sign << 5) | 0b0'11'111);
        }
        if (exp == 0 && frac == 0) {
            return static_cast<uint8_t>(sign << 5);  // ±0
        }

        int e_unbiased;
        float mant;
        f32_exp_frac_to_unbiased_and_mant(exp, frac, e_unbiased, mant);

        constexpr int bias = 1;  // E2M3 bias
        // 正规阈值：2^0 = 1
        if (exp == 0 || e_unbiased < 0) {
            float mag = (exp == 0) ? mant : std::ldexp(mant, e_unbiased);
            int m = rne_to_int(static_cast<double>(mag) * 8.0);  // 3-bit mantissa
            if (m <= 0) return static_cast<uint8_t>(sign << 5);
            if (m > 7) m = 7;
            return static_cast<uint8_t>((sign << 5) | static_cast<uint8_t>(m));
        }

        int e = e_unbiased;
        int m = rne_to_int(static_cast<double>(mant - 1.0f) * 8.0);
        if (m >= 8) { m = 0; ++e; }

        if (e > 2) {  // 最大指数 0b11 -> e_unbiased=2
            return static_cast<uint8_t>((sign << 5) | 0b0'11'111);
        }
        if (e < 0) {
            int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 3)));
            if (sub <= 0) return static_cast<uint8_t>(sign << 5);
            if (sub > 7) sub = 7;
            return static_cast<uint8_t>((sign << 5) | static_cast<uint8_t>(sub));
        }

        uint8_t e_bits = static_cast<uint8_t>(e + bias);
        return static_cast<uint8_t>((sign << 5) | (e_bits << 3) | (m & 0x7));
    }

    static float decode_to_float(uint8_t bits) {
        uint8_t sign = (bits >> 5) & 0x1u;
        uint8_t exp = (bits >> 3) & 0x3u;  // 2-bit exponent
        uint8_t mant = bits & 0x7u;         // 3-bit mantissa
        constexpr int bias = 1;

        if (exp == 0) {
            float v = static_cast<float>(mant) * (1.0f / 8.0f);  // 次正规
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
        int e = static_cast<int>(exp) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }

public:
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 1;
    static constexpr int kMantissaBits = 3;

    constexpr float6_e2m3fn() : rep_(0) {}

    explicit float6_e2m3fn(float f) : rep_(encode_from_float(f)) {}
    explicit float6_e2m3fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
    explicit float6_e2m3fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}

    constexpr uint8_t rep() const { return rep_; }

    static constexpr float6_e2m3fn FromRep(uint8_t rep) {
        return float6_e2m3fn(rep, ConstructFromRepTag{});
    }

    explicit operator float() const { return decode_to_float(rep_); }
    explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
    explicit operator bool() const { return (rep_ & 0x1Fu) != 0; }

    float6_e2m3fn operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x20u)); }

    float6_e2m3fn operator+(const float6_e2m3fn& other) const {
        return float6_e2m3fn(static_cast<float>(*this) + static_cast<float>(other));
    }
    float6_e2m3fn operator-(const float6_e2m3fn& other) const {
        return float6_e2m3fn(static_cast<float>(*this) - static_cast<float>(other));
    }
    float6_e2m3fn operator*(const float6_e2m3fn& other) const {
        return float6_e2m3fn(static_cast<float>(*this) * static_cast<float>(other));
    }
    float6_e2m3fn operator/(const float6_e2m3fn& other) const {
        return float6_e2m3fn(static_cast<float>(*this) / static_cast<float>(other));
    }

    bool operator==(const float6_e2m3fn& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
    bool operator!=(const float6_e2m3fn& other) const { return !(*this == other); }
    bool operator<(const float6_e2m3fn& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) return false;
        return a < b;
    }
    bool operator<=(const float6_e2m3fn& other) const { return *this < other || *this == other; }
    bool operator>(const float6_e2m3fn& other) const { return other < *this; }
    bool operator>=(const float6_e2m3fn& other) const { return other <= *this; }

    // ACL 枚举获取
    static constexpr aclDataType getACLenum() { return ACL_FLOAT6_E2M3; }
};

class float6_e3m2fn {
private:
    uint8_t rep_;
    struct ConstructFromRepTag {};
    constexpr float6_e3m2fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}

    static uint8_t encode_from_float(float f) {
        uint32_t u = bit_cast<uint32_t>(f);
        uint32_t sign = u >> 31;
        uint32_t exp = (u >> 23) & 0xFFu;
        uint32_t frac = u & 0x7FFFFFu;

        if (exp == 0xFFu) {
            return static_cast<uint8_t>((sign << 5) | 0b0'111'11);
        }
        if (exp == 0 && frac == 0) {
            return static_cast<uint8_t>(sign << 5);
        }

        int e_unbiased;
        float mant;
        f32_exp_frac_to_unbiased_and_mant(exp, frac, e_unbiased, mant);

        constexpr int bias = 3;  // E3M2 bias
        // 正规阈值：2^-2 = 0.25
        if (exp == 0 || e_unbiased < -2) {
            float mag = (exp == 0) ? mant : std::ldexp(mant, e_unbiased);
            int m = rne_to_int(static_cast<double>(mag) * 4.0);  // 2-bit mantissa
            if (m <= 0) return static_cast<uint8_t>(sign << 5);
            if (m > 3) m = 3;
            return static_cast<uint8_t>((sign << 5) | static_cast<uint8_t>(m));
        }

        int e = e_unbiased;
        int m = rne_to_int(static_cast<double>(mant - 1.0f) * 4.0);
        if (m >= 4) { m = 0; ++e; }

        if (e > 4) {  // 最大指数 0b111 -> e_unbiased=4
            return static_cast<uint8_t>((sign << 5) | 0b0'111'11);
        }
        if (e < -2) {
            int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 2)));
            if (sub <= 0) return static_cast<uint8_t>(sign << 5);
            if (sub > 3) sub = 3;
            return static_cast<uint8_t>((sign << 5) | static_cast<uint8_t>(sub));
        }

        uint8_t e_bits = static_cast<uint8_t>(e + bias);
        return static_cast<uint8_t>((sign << 5) | (e_bits << 2) | (m & 0x3));
    }

    static float decode_to_float(uint8_t bits) {
        uint8_t sign = (bits >> 5) & 0x1u;
        uint8_t exp = (bits >> 2) & 0x7u;  // 3-bit exponent
        uint8_t mant = bits & 0x3u;         // 2-bit mantissa
        constexpr int bias = 3;
        return mx_decode_sign_exp_mant(sign, exp, mant, bias, 0.25f);
    }

public:
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = 3;
    static constexpr int kMantissaBits = 2;

    constexpr float6_e3m2fn() : rep_(0) {}

    explicit float6_e3m2fn(float f) : rep_(encode_from_float(f)) {}
    explicit float6_e3m2fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
    explicit float6_e3m2fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}

    constexpr uint8_t rep() const { return rep_; }

    static constexpr float6_e3m2fn FromRep(uint8_t rep) {
        return float6_e3m2fn(rep, ConstructFromRepTag{});
    }

    explicit operator float() const { return decode_to_float(rep_); }
    explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
    explicit operator bool() const { return (rep_ & 0x1Fu) != 0; }

    float6_e3m2fn operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x20u)); }

    float6_e3m2fn operator+(const float6_e3m2fn& other) const {
        return float6_e3m2fn(static_cast<float>(*this) + static_cast<float>(other));
    }
    float6_e3m2fn operator-(const float6_e3m2fn& other) const {
        return float6_e3m2fn(static_cast<float>(*this) - static_cast<float>(other));
    }
    float6_e3m2fn operator*(const float6_e3m2fn& other) const {
        return float6_e3m2fn(static_cast<float>(*this) * static_cast<float>(other));
    }
    float6_e3m2fn operator/(const float6_e3m2fn& other) const {
        return float6_e3m2fn(static_cast<float>(*this) / static_cast<float>(other));
    }

    bool operator==(const float6_e3m2fn& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
    bool operator!=(const float6_e3m2fn& other) const { return !(*this == other); }
    bool operator<(const float6_e3m2fn& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) return false;
        return a < b;
    }
    bool operator<=(const float6_e3m2fn& other) const { return *this < other || *this == other; }
    bool operator>(const float6_e3m2fn& other) const { return other < *this; }
    bool operator>=(const float6_e3m2fn& other) const { return other <= *this; }

    // ACL 枚举获取
    static constexpr aclDataType getACLenum() { return ACL_FLOAT6_E3M2; }
};

}  // namespace dtypes
}  // namespace asnumpy

