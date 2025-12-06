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
 class float6_e2m3fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float6_e2m3fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
public:
    // 处理特殊值（Inf/NaN/Zero）
    static uint8_t encode_special_values(uint32_t sign, uint32_t exp, uint32_t frac) {
        if (exp == constants::kFloat32MaxExponent) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat6E2M3FnSignShift) | static_cast<uint8_t>(0b0'11'111));
        }
        if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat6E2M3FnSignShift);
        }
        return static_cast<uint8_t>(0);  // 非特殊值
    }

    // 编码次正规数
    static uint8_t encode_subnormal(uint32_t sign, float mant, int e_unbiased) {
        float mag = (e_unbiased == constants::kFloat32SubnormalExponent) ? mant : std::ldexp(mant, e_unbiased);
        int m = rne_to_int(static_cast<double>(mag) * constants::kFloat6E2M3FnSubnormalGrid);
        if (m <= 0) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat6E2M3FnSignShift);
        }
        if (m > constants::kFloat6E2M3FnMaxMantissa) {
            m = constants::kFloat6E2M3FnMaxMantissa;
        }
        // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat6E2M3FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
    }

    // 编码正规数
    static uint8_t encode_normal(uint32_t sign, int e, int m, int bias, float mant) {
        return encode_normal_impl<constants::kFloat6E2M3FnMaxExponent,
            constants::kFloat6E2M3FnSubnormalThreshold,
            constants::kFloat6E2M3FnMaxMantissa,
            constants::kFloat6E2M3FnSubnormalLdexpOffset,
            constants::kFloat6E2M3FnSignShift,
            constants::kFloat6E2M3FnExponentShift,
            constants::kFloat6E2M3FnMantissaMask,
            static_cast<uint8_t>(0b0'11'111)>(sign, e, m, bias, mant);
    }

    static uint8_t encode_from_float(float f) {
        return encode_from_float_helper<float6_e2m3fn,
            constants::kFloat6E2M3FnBias,
            constants::kFloat6E2M3FnSubnormalThreshold,
            constants::kFloat6E2M3FnMantissaQuantization>(f);
     }

private:
    static float decode_to_float(uint8_t bits) {
        uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat6E2M3FnSignShift) & static_cast<uint8_t>(0x1u));
        uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat6E2M3FnExponentShift) & constants::kFloat6E2M3FnExponentMask);  // 2-bit exponent
        uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat6E2M3FnMantissaMask);         // 3-bit mantissa
        constexpr int bias = constants::kFloat6E2M3FnBias;

        if (exp == static_cast<uint8_t>(0)) {
            float v = static_cast<float>(mant) * (1.0f / static_cast<float>(constants::kFloat6E2M3FnSubnormalGrid));  // 次正规
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat6E2M3FnMantissaStep);
        // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }

public:
    static constexpr int kBits = 6;
    static constexpr int kExponentBias = constants::kFloat6E2M3FnBias;
    static constexpr int kMantissaBits = constants::kFloat6E2M3FnMantissaBits;
 
     constexpr float6_e2m3fn() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float6_e2m3fn(float f) : rep_(encode_from_float(f)) {}
     explicit float6_e2m3fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float6_e2m3fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float6_e2m3fn FromRep(uint8_t rep) {
         return float6_e2m3fn(rep, ConstructFromRepTag{});
     }
 
     FLOAT_TYPE_CONVERSION_OPERATORS(float6_e2m3fn, decode_to_float, constants::kFloat6MantissaMask5)

     float6_e2m3fn operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat6SignBitMask));
     }
 
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
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float6_e2m3fn& other) const {
         return !(*this == other);
     }
    bool operator<(const float6_e2m3fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float6_e2m3fn& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float6_e2m3fn& other) const {
        return other < *this;
    }
    bool operator>=(const float6_e2m3fn& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT6_E2M3;
     }
 };


 class float6_e3m2fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float6_e3m2fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
public:
     // 处理特殊值（Inf/NaN/Zero）
     static uint8_t encode_special_values(uint32_t sign, uint32_t exp, uint32_t frac) {
         if (exp == constants::kFloat32MaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat6E3M2FnSignShift) | static_cast<uint8_t>(0b0'111'11));
         }
         if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat6E3M2FnSignShift);
         }
         return static_cast<uint8_t>(0);  // 非特殊值
     }

     // 编码次正规数
     static uint8_t encode_subnormal(uint32_t sign, float mant, int e_unbiased) {
         float mag = (e_unbiased == constants::kFloat32SubnormalExponent) ? mant : std::ldexp(mant, e_unbiased);
         int m = rne_to_int(static_cast<double>(mag) * constants::kFloat6E3M2FnSubnormalGrid);
         if (m <= 0) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat6E3M2FnSignShift);
         }
         if (m > constants::kFloat6E3M2FnMaxMantissa) {
             m = constants::kFloat6E3M2FnMaxMantissa;
         }
         // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
         return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat6E3M2FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
     }

     // 编码正规数
     static uint8_t encode_normal(uint32_t sign, int e, int m, int bias, float mant) {
         return encode_normal_impl<constants::kFloat6E3M2FnMaxExponent,
             constants::kFloat6E3M2FnSubnormalThreshold,
             constants::kFloat6E3M2FnMaxMantissa,
             constants::kFloat6E3M2FnSubnormalLdexpOffset,
             constants::kFloat6E3M2FnSignShift,
             constants::kFloat6E3M2FnExponentShift,
             constants::kFloat6E3M2FnMantissaMask,
             static_cast<uint8_t>(0b0'111'11)>(sign, e, m, bias, mant);
     }

     static uint8_t encode_from_float(float f) {
         return encode_from_float_helper<float6_e3m2fn,
             constants::kFloat6E3M2FnBias,
             constants::kFloat6E3M2FnSubnormalThreshold,
             constants::kFloat6E3M2FnMantissaQuantization>(f);
     }

private:
     static float decode_to_float(uint8_t bits) {
         uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat6E3M2FnSignShift) & static_cast<uint8_t>(0x1u));
         uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat6E3M2FnExponentShift) & constants::kFloat6E3M2FnExponentMask);  // 3-bit exponent
         uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat6E3M2FnMantissaMask);         // 2-bit mantissa
         constexpr int bias = constants::kFloat6E3M2FnBias;

         if (exp == static_cast<uint8_t>(0)) {
             float v = static_cast<float>(mant) * (1.0f / static_cast<float>(constants::kFloat6E3M2FnSubnormalGrid));  // 次正规
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat6E3M2FnMantissaStep);
         // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
     }

 public:
     static constexpr int kBits = 6;
     static constexpr int kExponentBias = constants::kFloat6E3M2FnBias;
     static constexpr int kMantissaBits = constants::kFloat6E3M2FnMantissaBits;
 
     constexpr float6_e3m2fn() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float6_e3m2fn(float f) : rep_(encode_from_float(f)) {}
     explicit float6_e3m2fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float6_e3m2fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float6_e3m2fn FromRep(uint8_t rep) {
         return float6_e3m2fn(rep, ConstructFromRepTag{});
     }

     FLOAT_TYPE_CONVERSION_OPERATORS(float6_e3m2fn, decode_to_float, constants::kFloat6MantissaMask5)

     float6_e3m2fn operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat6SignBitMask));
     }
 
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
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float6_e3m2fn& other) const {
         return !(*this == other);
     }
    bool operator<(const float6_e3m2fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float6_e3m2fn& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float6_e3m2fn& other) const {
        return other < *this;
    }
    bool operator>=(const float6_e3m2fn& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT6_E3M2;
     }
 };

}  // namespace dtypes
}  // namespace asnumpy
