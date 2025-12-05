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

 class float4_e2m1fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float4_e2m1fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     // 处理特殊值（Inf/NaN/Zero）
     static uint8_t encode_special_values(uint32_t sign, uint32_t exp, uint32_t frac) {
         if (exp == constants::kFloat32MaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift) | static_cast<uint8_t>(0b0'11'1));
         }
        if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift);
        }
         return static_cast<uint8_t>(0);  // 非特殊值
     }

     // 编码次正规数
     static uint8_t encode_subnormal(uint32_t sign, float mant, int e_unbiased) {
         float mag = (e_unbiased == constants::kFloat32SubnormalExponent) ? mant : std::ldexp(mant, e_unbiased);
         int m = rne_to_int(static_cast<double>(mag) * constants::kFloat4E2M1FnSubnormalGrid);
         if (m <= 0) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift);
         }
         if (m > constants::kFloat4E2M1FnMaxMantissa) {
             m = constants::kFloat4E2M1FnMaxMantissa;
         }
         // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
         return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
     }

     // 编码正规数
     static uint8_t encode_normal(uint32_t sign, int e, int m, int bias, float mant) {
         if (e > constants::kFloat4E2M1FnMaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift) | static_cast<uint8_t>(0b0'11'1));
         }
         if (e < constants::kFloat4E2M1FnSubnormalThreshold) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + constants::kFloat4E2M1FnSubnormalLdexpOffset)));
             if (sub <= 0) {
                 // 显式转换：从 uint32_t 到 uint8_t
                 return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift);
             }
             if (sub > constants::kFloat4E2M1FnMaxMantissa) {
                 sub = constants::kFloat4E2M1FnMaxMantissa;
             }
             // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(sub)));
         }
         // 显式转换：从 int 到 uint8_t（改变符号）
        uint8_t e_bits = static_cast<uint8_t>(static_cast<unsigned int>(e + bias));
        // 显式转换：从 uint32_t 到 uint8_t，从 int 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E2M1FnSignShift) | static_cast<uint8_t>(e_bits << constants::kFloat4E2M1FnExponentShift) | static_cast<uint8_t>(static_cast<unsigned int>(m) & static_cast<unsigned int>(constants::kFloat4E2M1FnMantissaMask)));
     }

     static uint8_t encode_from_float(float f) {
         return encode_from_float_impl<constants::kFloat4E2M1FnBias,
             constants::kFloat4E2M1FnSubnormalThreshold,
             constants::kFloat4E2M1FnMantissaQuantization>(
             f,
             [](uint32_t s, uint32_t e, uint32_t fr) {
                 return encode_special_values(s, e, fr);
             },
             [](uint32_t s, float m, int e) {
                 return encode_subnormal(s, m, e);
             },
             [](uint32_t s, int e, int m, int b, float mant) {
                 return encode_normal(s, e, m, b, mant);
             });
     }
 
    static float decode_to_float(uint8_t bits) {
        uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat4E2M1FnSignShift) & static_cast<uint8_t>(0x1u));
        uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat4E2M1FnExponentShift) & constants::kFloat4E2M1FnExponentMask);  // 2-bit exponent
        uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat4E2M1FnMantissaMask);         // 1-bit mantissa
        constexpr int bias = constants::kFloat4E2M1FnBias;

        if (exp == static_cast<uint8_t>(0)) {
            float v = static_cast<float>(mant) * static_cast<float>(constants::kFloat4E2M1FnMantissaStep);  // 次正规
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat4E2M1FnMantissaStep);
        // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }

public:
    static constexpr int kBits = 4;
    static constexpr int kExponentBias = constants::kFloat4E2M1FnBias;
    static constexpr int kMantissaBits = constants::kFloat4E2M1FnMantissaBits;
 
     constexpr float4_e2m1fn() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float4_e2m1fn(float f) : rep_(encode_from_float(f)) {}
     explicit float4_e2m1fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float4_e2m1fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float4_e2m1fn FromRep(uint8_t rep) {
         return float4_e2m1fn(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { 
        return decode_to_float(rep_); 
    }
     explicit operator double() const {
         return FloatConversionOps<&float4_e2m1fn::decode_to_float, constants::kFloat4MantissaMask3>::to_double(rep_);
    }
     explicit operator bool() const {
         return FloatConversionOps<&float4_e2m1fn::decode_to_float, constants::kFloat4MantissaMask3>::to_bool(rep_);
    }

     float4_e2m1fn operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat4SignBitMask)); 
    }
 
     float4_e2m1fn operator+(const float4_e2m1fn& other) const {
         return float4_e2m1fn(static_cast<float>(*this) + static_cast<float>(other));
     }
     float4_e2m1fn operator-(const float4_e2m1fn& other) const {
         return float4_e2m1fn(static_cast<float>(*this) - static_cast<float>(other));
     }
     float4_e2m1fn operator*(const float4_e2m1fn& other) const {
         return float4_e2m1fn(static_cast<float>(*this) * static_cast<float>(other));
     }
     float4_e2m1fn operator/(const float4_e2m1fn& other) const {
         return float4_e2m1fn(static_cast<float>(*this) / static_cast<float>(other));
     }
 
    bool operator==(const float4_e2m1fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float4_e2m1fn& other) const {
         return !(*this == other);
     }
    bool operator<(const float4_e2m1fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float4_e2m1fn& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float4_e2m1fn& other) const {
        return other < *this;
    }
    bool operator>=(const float4_e2m1fn& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT4_E2M1; 
    }
 };


 class float4_e1m2fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float4_e1m2fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     // 处理特殊值（Inf/NaN/Zero）
     static uint8_t encode_special_values(uint32_t sign, uint32_t exp, uint32_t frac) {
         if (exp == constants::kFloat32MaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift) | static_cast<uint8_t>(0b0'1'11));
         }
        if (exp == static_cast<uint32_t>(0) && frac == static_cast<uint32_t>(0)) {
            // 显式转换：从 uint32_t 到 uint8_t
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift);
        }
         return static_cast<uint8_t>(0);  // 非特殊值
     }

     // 编码次正规数
     static uint8_t encode_subnormal(uint32_t sign, float mant, int e_unbiased) {
         float mag = (e_unbiased == constants::kFloat32SubnormalExponent) ? mant : std::ldexp(mant, e_unbiased);
         int m = rne_to_int(static_cast<double>(mag) * constants::kFloat4E1M2FnSubnormalGrid);
         if (m <= 0) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift);
         }
         if (m > constants::kFloat4E1M2FnMaxMantissa) {
             m = constants::kFloat4E1M2FnMaxMantissa;
         }
         // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
         return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(m)));
     }

     // 编码正规数
     static uint8_t encode_normal(uint32_t sign, int e, int m, int bias, float mant) {
         if (e > constants::kFloat4E1M2FnMaxExponent) {
             // 显式转换：从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift) | static_cast<uint8_t>(0b0'1'11));
         }
         if (e < constants::kFloat4E1M2FnSubnormalThreshold) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + constants::kFloat4E1M2FnSubnormalLdexpOffset)));
             if (sub <= 0) {
                 // 显式转换：从 uint32_t 到 uint8_t
                 return static_cast<uint8_t>(static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift);
             }
             if (sub > constants::kFloat4E1M2FnMaxMantissa) {
                 sub = constants::kFloat4E1M2FnMaxMantissa;
             }
             // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
             return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift) | static_cast<uint8_t>(static_cast<unsigned int>(sub)));
         }
         // 显式转换：从 int 到 uint8_t（改变符号）
        uint8_t e_bits = static_cast<uint8_t>(static_cast<unsigned int>(e + bias));
        // 显式转换：从 uint32_t 到 uint8_t，从 int 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << constants::kFloat4E1M2FnSignShift) | static_cast<uint8_t>(e_bits << constants::kFloat4E1M2FnExponentShift) | static_cast<uint8_t>(static_cast<unsigned int>(m) & static_cast<unsigned int>(constants::kFloat4E1M2FnMantissaMask)));
     }

     static uint8_t encode_from_float(float f) {
         return encode_from_float_impl<constants::kFloat4E1M2FnBias,
             constants::kFloat4E1M2FnSubnormalThreshold,
             constants::kFloat4E1M2FnMantissaQuantization>(
             f,
             [](uint32_t s, uint32_t e, uint32_t fr) {
                 return encode_special_values(s, e, fr);
             },
             [](uint32_t s, float m, int e) {
                 return encode_subnormal(s, m, e);
             },
             [](uint32_t s, int e, int m, int b, float mant) {
                 return encode_normal(s, e, m, b, mant);
             });
     }
 
    static float decode_to_float(uint8_t bits) {
        uint8_t sign = static_cast<uint8_t>((bits >> constants::kFloat4E1M2FnSignShift) & static_cast<uint8_t>(0x1u));
        uint8_t exp = static_cast<uint8_t>((bits >> constants::kFloat4E1M2FnExponentShift) & constants::kFloat4E1M2FnExponentMask);  // 1-bit exponent
        uint8_t mant = static_cast<uint8_t>(bits & constants::kFloat4E1M2FnMantissaMask);         // 2-bit mantissa
        constexpr int bias = constants::kFloat4E1M2FnBias;

        if (exp == static_cast<uint8_t>(0)) {
            float v = static_cast<float>(mant) * (1.0f / static_cast<float>(constants::kFloat4E1M2FnSubnormalGrid));  // 次正规
            return sign ? -v : v;
        }
        float base = 1.0f + static_cast<float>(mant) * static_cast<float>(constants::kFloat4E1M2FnMantissaStep);
        // 显式转换：从 uint8_t 到 int（改变符号，但需要用于有符号运算）
        int e = static_cast<int>(static_cast<int32_t>(exp)) - bias;
        float v = std::ldexp(base, e);
        return sign ? -v : v;
    }

public:
    static constexpr int kBits = 4;
    static constexpr int kExponentBias = constants::kFloat4E1M2FnBias;
    static constexpr int kMantissaBits = constants::kFloat4E1M2FnMantissaBits;
 
     constexpr float4_e1m2fn() : rep_(static_cast<uint8_t>(0)) {}
 
     explicit float4_e1m2fn(float f) : rep_(encode_from_float(f)) {}
     explicit float4_e1m2fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float4_e1m2fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const {
         return rep_;
     }
 
     static constexpr float4_e1m2fn FromRep(uint8_t rep) {
         return float4_e1m2fn(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { 
        return decode_to_float(rep_); 
    }
     explicit operator double() const { 
        return FloatConversionOps<&float4_e1m2fn::decode_to_float, constants::kFloat4MantissaMask3>::to_double(rep_);
    }
     explicit operator bool() const {
         return FloatConversionOps<&float4_e1m2fn::decode_to_float, constants::kFloat4MantissaMask3>::to_bool(rep_);
    }

     float4_e1m2fn operator-() const {
         return FromRep(static_cast<uint8_t>(rep_ ^ constants::kFloat4SignBitMask));
     }
 
     float4_e1m2fn operator+(const float4_e1m2fn& other) const {
         return float4_e1m2fn(static_cast<float>(*this) + static_cast<float>(other));
     }
     float4_e1m2fn operator-(const float4_e1m2fn& other) const {
         return float4_e1m2fn(static_cast<float>(*this) - static_cast<float>(other));
     }
     float4_e1m2fn operator*(const float4_e1m2fn& other) const {
         return float4_e1m2fn(static_cast<float>(*this) * static_cast<float>(other));
     }
     float4_e1m2fn operator/(const float4_e1m2fn& other) const {
         return float4_e1m2fn(static_cast<float>(*this) / static_cast<float>(other));
     }
 
    bool operator==(const float4_e1m2fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
     bool operator!=(const float4_e1m2fn& other) const { 
        return !(*this == other); 
    }
    bool operator<(const float4_e1m2fn& other) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) {
            return false;
        }
        return a < b;
    }
    bool operator<=(const float4_e1m2fn& other) const {
        return *this < other || *this == other;
    }
    bool operator>(const float4_e1m2fn& other) const {
        return other < *this;
    }
    bool operator>=(const float4_e1m2fn& other) const {
        return other <= *this;
    }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() {
         return ACL_FLOAT4_E1M2;
     }
 };

}  // namespace dtypes
}  // namespace asnumpy
