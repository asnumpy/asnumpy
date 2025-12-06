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

#include <cstdint>
#include <cmath>
#include <limits>
#include <acl/acl.h>
#include "float_constants.hpp"
#include "float_utils.hpp"

namespace asnumpy {
namespace dtypes {
 class float8_e5m2 {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e5m2(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
public:
    // 处理特殊值（Inf/NaN/Zero）
    static uint8_t encode_special_values(uint32_t sign, uint32_t exp, uint32_t frac) {
        if (exp == constants::kFloat32MaxExponent) {
            // Inf/NaN
            if (frac == static_cast<uint32_t>(0)) {
                // 显式转换：从 uint32_t 到 uint8_t
                return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E5M2SignShift) | static_cast<uint8_t>(0b0'11111'00));
            }
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E5M2SignShift) | static_cast<uint8_t>(0b0'11111'10));  // qNaN
        }
        if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
            // Preserve ±0
            // 显式转换：从 uint32_t 到 uint8_t（改变大小，但符号不变）
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat8E5M2SignShift);
        }
        return static_cast<uint8_t>(0);  // 非特殊值
    }

    // 编码次正规数
    static uint8_t encode_subnormal(uint32_t sign, float a, int e_unbiased) {
        float mag = (e_unbiased == constants::kFloat32SubnormalExponent) ? a : std::ldexp(a, e_unbiased);
        int m = rne_to_int(static_cast<double>(mag) * constants::kFloat8E5M2SubnormalGrid);
        if (m <= 0) {
            // 显式转换：从 uint32_t 到 uint8_t（改变大小，但符号不变）
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat8E5M2SignShift);
        }
        if (m > constants::kFloat8E5M2MaxMantissa) {
            m = constants::kFloat8E5M2MaxMantissa;
        }
        // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E5M2SignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
    }

    // 编码正规数
    static uint8_t encode_normal(uint32_t sign, int e, int m, int bias, float mant) {
        return encode_normal_impl<constants::kFloat8E5M2MaxExponent,
            constants::kFloat8E5M2SubnormalThreshold,
            constants::kFloat8E5M2MaxMantissa,
            constants::kFloat8E5M2SubnormalLdexpOffset,
            constants::kFloat8E5M2SignShift,
            constants::kFloat8E5M2ExponentShift,
            constants::kFloat8E5M2MantissaMask,
            static_cast<uint8_t>(0b0'11111'00)>(sign, e, m, bias, mant);
    }

    static uint8_t encode_from_float(float f) {
        return encode_from_float_helper<float8_e5m2,
            constants::kFloat8E5M2Bias,
            constants::kFloat8E5M2SubnormalThreshold,
            constants::kFloat8E5M2MantissaQuantization>(f);
     }

private:
    static float decode_to_float(uint8_t bits) {
        uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat8E5M2SignShift) & static_cast<uint8_t>(0x1u));
        uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat8E5M2ExponentShift) & constants::kFloat8E5M2ExponentMask);
        uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat8E5M2MantissaMask);
        constexpr int bias = constants::kFloat8E5M2Bias;

        if (static_cast<unsigned int>(exp) == static_cast<unsigned int>(constants::kFloat8E5M2MaxExponentValue)) {
            if (mant == static_cast<uint8_t>(0)) {
                return sign ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
            }
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (exp == static_cast<uint8_t>(0)) {
            float v = static_cast<float>(mant) * (1.0f / static_cast<float>(constants::kFloat8E5M2SubnormalGrid));  // 2^-16
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat8E5M2MantissaStep);
        // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }
 
public:
    static constexpr int kBits = 8;
    static constexpr int kExponentBias = constants::kFloat8E5M2Bias;
    static constexpr int kMantissaBits = constants::kFloat8E5M2MantissaBits;
 
     constexpr float8_e5m2() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float8_e5m2(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e5m2(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e5m2(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_; 
    }
 
     static constexpr float8_e5m2 FromRep(uint8_t rep) {
         return float8_e5m2(rep, ConstructFromRepTag{});
     }
 
     FLOAT_TYPE_CONVERSION_OPERATORS(float8_e5m2, decode_to_float, constants::kFloat8MantissaMask7)

     float8_e5m2 operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat8SignBitMask));
     }
 
     float8_e5m2 operator+(const float8_e5m2& other) const {
         return float8_e5m2(static_cast<float>(*this) + static_cast<float>(other));
     }
     float8_e5m2 operator-(const float8_e5m2& other) const {
         return float8_e5m2(static_cast<float>(*this) - static_cast<float>(other));
     }
     float8_e5m2 operator*(const float8_e5m2& other) const {
         return float8_e5m2(static_cast<float>(*this) * static_cast<float>(other));
     }
     float8_e5m2 operator/(const float8_e5m2& other) const {
         return float8_e5m2(static_cast<float>(*this) / static_cast<float>(other));
     }
 
    bool operator==(const float8_e5m2& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float8_e5m2& other) const {
         return !(*this == other);
     }
    bool operator<(const float8_e5m2& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float8_e5m2& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float8_e5m2& other) const {
        return other < *this;
    }
    bool operator>=(const float8_e5m2& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT8_E5M2;
     }
 };


 class float8_e4m3fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e4m3fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
    static uint8_t encode_from_float(float f) {
        Float32Components components = extract_float32_components(f);
        uint32_t sign = components.sign;
        uint32_t exp = components.exp;
        uint32_t frac = components.frac;

        // 无 Inf（finite-only），Inf/NaN 统一编码为外层 NaN
        if (exp == constants::kFloat32MaxExponent) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(0b0'1111'111));
        }
        if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift);  // 保留 ±0
        }

        Float32Normalized normalized = normalize_float32_components(exp, frac);
        int e_unbiased = normalized.e_unbiased;
        float mant = normalized.mant;

        constexpr int bias = constants::kFloat8E4M3FnBias;
         // e4m3fn 正规阈值：2^-6
         if (exp == static_cast<uint32_t>(0) || e_unbiased < constants::kFloat8E4M3FnSubnormalThreshold) {
            float mag = (exp == static_cast<uint32_t>(0)) ? mant : std::ldexp(mant, e_unbiased);
            int m = rne_to_int(static_cast<double>(mag) * constants::kFloat8E4M3FnSubnormalGrid);  // 2^9 网格
            if (m <= 0) {
                // 显式转换：从 uint32_t 到 uint8_t
                return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift);
            }
             if (m > constants::kFloat8E4M3FnMaxMantissa) {
                 m = constants::kFloat8E4M3FnMaxMantissa;
             }
             // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
         }

        int e = e_unbiased;
        int m = rne_to_int(static_cast<double>(mant - 1.0f) * static_cast<double>(constants::kFloat8E4M3FnMantissaQuantization));
        if (m >= constants::kFloat8E4M3FnMantissaQuantization) {
            m = 0;
            ++e;
        }

        // 最大指数 e_unbiased=8（exp_bits=0x0F）范围内保留外层 NaN
         if (e > constants::kFloat8E4M3FnMaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(0b0'1111'111));
         }
        if (e < constants::kFloat8E4M3FnSubnormalThreshold) {
            int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + constants::kFloat8E4M3FnSubnormalLdexpOffset)));
            if (sub <= 0) {
                // 显式转换：从 uint32_t 到 uint8_t
                return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift);
            }
             if (sub > constants::kFloat8E4M3FnMaxMantissa) {
                 sub = constants::kFloat8E4M3FnMaxMantissa;
             }
             // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(sub)));
         }

         // 显式转换：从 int 到 uint8_t（改变符号）
         uint8_t e_bits = static_cast<uint8_t>(static_cast<unsigned int>(e + bias));
         if (static_cast<unsigned int>(e_bits) == static_cast<unsigned int>(constants::kFloat8E4M3FnMaxExponentValue) && static_cast<unsigned int>(static_cast<unsigned int>(m) & static_cast<unsigned int>(constants::kFloat8E4M3FnMantissaMask)) == static_cast<unsigned int>(constants::kFloat8E4M3FnMaxMantissa)) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(0b0'1111'111));
         }
         // 显式转换：从 uint32_t 到 uint8_t，从 int 到 uint8_t
         return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat8E4M3FnSignShift) | static_cast<uint8_t>(e_bits << constants::kFloat8E4M3FnExponentShift) | static_cast<uint8_t>(static_cast<unsigned int>(m) & static_cast<unsigned int>(constants::kFloat8E4M3FnMantissaMask)));
     }
 
    static float decode_to_float(uint8_t bits) {
        uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat8E4M3FnSignShift) & static_cast<uint8_t>(0x1u));
        uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat8E4M3FnExponentShift) & constants::kFloat8E4M3FnExponentMask);
        uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat8E4M3FnMantissaMask);
        constexpr int bias = constants::kFloat8E4M3FnBias;

        // 外层 NaN
        if (static_cast<unsigned int>(exp) == static_cast<unsigned int>(constants::kFloat8E4M3FnMaxExponentValue) && static_cast<int>(mant) == constants::kFloat8E4M3FnMaxMantissa) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        if (exp == static_cast<uint8_t>(0)) {
            float v = static_cast<float>(mant) * (1.0f / static_cast<float>(constants::kFloat8E4M3FnSubnormalGrid));  // 2^-9
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat8E4M3FnMantissaStep);
        // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }
 
public:
    static constexpr int kBits = 8;
    static constexpr int kExponentBias = constants::kFloat8E4M3FnBias;
    static constexpr int kMantissaBits = constants::kFloat8E4M3FnMantissaBits;
 
     constexpr float8_e4m3fn() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float8_e4m3fn(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e4m3fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e4m3fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float8_e4m3fn FromRep(uint8_t rep) {
         return float8_e4m3fn(rep, ConstructFromRepTag{});
     }

     FLOAT_TYPE_CONVERSION_OPERATORS(float8_e4m3fn, decode_to_float, constants::kFloat8MantissaMask7)

     float8_e4m3fn operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat8SignBitMask));
     }
 
     float8_e4m3fn operator+(const float8_e4m3fn& other) const {
         return float8_e4m3fn(static_cast<float>(*this) + static_cast<float>(other));
     }
     float8_e4m3fn operator-(const float8_e4m3fn& other) const {
         return float8_e4m3fn(static_cast<float>(*this) - static_cast<float>(other));
     }
     float8_e4m3fn operator*(const float8_e4m3fn& other) const {
         return float8_e4m3fn(static_cast<float>(*this) * static_cast<float>(other));
     }
     float8_e4m3fn operator/(const float8_e4m3fn& other) const {
         return float8_e4m3fn(static_cast<float>(*this) / static_cast<float>(other));
     }
 
    bool operator==(const float8_e4m3fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float8_e4m3fn& other) const {
         return !(*this == other);
     }
    bool operator<(const float8_e4m3fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float8_e4m3fn& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float8_e4m3fn& other) const {
        return other < *this;
    }
    bool operator>=(const float8_e4m3fn& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT8_E4M3FN;
     }
 };


 class float8_e8m0 {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e8m0(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
    static uint8_t encode_from_float(float f) {
        if (std::isnan(f)) {
            return constants::kFloat8E8M0NaNValue;
        }
        if (f < 0.0f) {
            return constants::kFloat8E8M0NaNValue;  // 无符号：负值->NaN
        }
        if (std::isinf(f)) {
            return constants::kFloat8E8M0InfValue;  // 无 Inf，饱和到最大有限
        }
        if (f == 0.0f) {
            return constants::kFloat8E8M0ZeroValue;      // 无 0 语义，映射到最小有限
        }

         int e;
         float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5,1)
         // 最近 2^k：阈值为 sqrt(2)/2 ≈ 0.7071
        int k = (m < constants::kFloat8E8M0RoundingThreshold) ? (e - 1) : e;
        int code = k + constants::kFloat8E8M0Bias;  // 偏置 127
        if (code < constants::kFloat8E8M0MinCode) {
            code = constants::kFloat8E8M0MinCode;
        }
        if (code > constants::kFloat8E8M0MaxCode) {
            code = constants::kFloat8E8M0MaxCode;
        }
        return static_cast<uint8_t>(code);
     }

     static float decode_to_float(uint8_t bits) {
         if (bits == constants::kFloat8E8M0NaNValue) {
             return std::numeric_limits<float>::quiet_NaN();
         }
         int e = static_cast<int>(bits) - constants::kFloat8E8M0Bias;
         return std::ldexp(1.0f, e);
     }
 
public:
    static constexpr int kBits = 8;
    static constexpr int kExponentBias = constants::kFloat8E8M0Bias;
    static constexpr int kMantissaBits = 0;
 
     constexpr float8_e8m0() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float8_e8m0(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e8m0(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e8m0(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float8_e8m0 FromRep(uint8_t rep) {
         return float8_e8m0(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const {
         return decode_to_float(rep_);
     }
     explicit operator double() const {
         return static_cast<double>(static_cast<float>(*this));
     }
     explicit operator bool() const {
         return true;  // 无 0 概念
     }
 
     float8_e8m0 operator-() const {
         return FromRep(constants::kFloat8E8M0NaNValue);  // 负号 -> NaN
     }
 
     float8_e8m0 operator+(const float8_e8m0& other) const {
         return float8_e8m0(static_cast<float>(*this) + static_cast<float>(other));
     }
     float8_e8m0 operator-(const float8_e8m0& other) const {
         return float8_e8m0(static_cast<float>(*this) - static_cast<float>(other));
     }
     float8_e8m0 operator*(const float8_e8m0& other) const {
         return float8_e8m0(static_cast<float>(*this) * static_cast<float>(other));
     }
     float8_e8m0 operator/(const float8_e8m0& other) const {
         return float8_e8m0(static_cast<float>(*this) / static_cast<float>(other));
     }
 
    bool operator==(const float8_e8m0& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float8_e8m0& other) const {
         return !(*this == other);
     }
    bool operator<(const float8_e8m0& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float8_e8m0& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float8_e8m0& other) const {
        return other < *this;
    }
    bool operator>=(const float8_e8m0& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT8_E8M0;
     }
 };

}  // namespace dtypes
}  // namespace asnumpy
