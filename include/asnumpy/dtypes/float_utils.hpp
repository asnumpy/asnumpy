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
#include <type_traits>
#include <bit>
#include "float_constants.hpp"

namespace asnumpy {
namespace dtypes {

// 使用 C++20 的 std::bit_cast 实现类型安全的位转换
// std::bit_cast 是类型安全的，不需要使用不安全的 memcpy
 template <class To, class From>
 inline auto bit_cast(const From& src) -> To {
     static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
     static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");
     static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
     return std::bit_cast<To>(src);
 }

// 从 float32 提取符号、指数和尾数组件
struct Float32Components {
    uint32_t sign;
    uint32_t exp;
    uint32_t frac;
};

inline Float32Components extract_float32_components(float f) {
    uint32_t u = bit_cast<uint32_t>(f);
    Float32Components components;
    components.sign = u >> constants::kFloat32SignShift;
    components.exp = (u >> constants::kFloat32ExponentShift) & constants::kFloat32ExponentMask;
    components.frac = u & constants::kFloat32MantissaMask;
    return components;
}

// 从 float32 组件计算无偏指数和尾数
struct Float32Normalized {
    int e_unbiased;
    float mant;
};

inline Float32Normalized normalize_float32_components(uint32_t exp, uint32_t frac) {
    Float32Normalized result;
    if (exp == static_cast<uint32_t>(0)) {
        result.e_unbiased = constants::kFloat32SubnormalExponent;
        result.mant = std::ldexp(static_cast<float>(frac), constants::kFloat32SubnormalLdexpOffset);
    } else {
        // 显式转换：从 uint32_t 到 int（改变符号，但需要用于有符号运算）
        result.e_unbiased = static_cast<int>(static_cast<int32_t>(exp)) - constants::kFloat32ExponentBias;
        result.mant = 1.0f + static_cast<float>(frac) * (1.0f / static_cast<float>(constants::kFloat32MantissaScale));
    }
    return result;
}
 
inline int rne_to_int(double x) {
    double f = std::floor(x);
    double frac = x - f;
    if (frac > constants::kRoundingThreshold) {
        return static_cast<int>(f) + 1;
    }
    if (frac < constants::kRoundingThreshold) {
        return static_cast<int>(f);
    }
    // ties to even
    return (static_cast<long long>(f) & 1LL) ? static_cast<int>(f) + 1
                                             : static_cast<int>(f);
}

// 浮点类型转换操作符的辅助函数模板
// 用于消除不同浮点类型类中重复的类型转换操作符实现
template<float (*DecodeToFloat)(uint8_t), uint8_t MantissaMask>
struct FloatConversionOps {
    static double to_double(uint8_t rep) {
        return static_cast<double>(DecodeToFloat(rep));
    }
    
    static bool to_bool(uint8_t rep) {
        return static_cast<uint8_t>(rep & MantissaMask) != static_cast<uint8_t>(0);
    }
};

// 编码正规数的通用模板函数
// 用于消除不同浮点类型类中重复的 encode_normal 实现
template<int MaxExponent, int SubnormalThreshold, int MaxMantissa, int SubnormalLdexpOffset,
         uint8_t SignShift, uint8_t ExponentShift, uint8_t MantissaMask, uint8_t InfValue>
inline uint8_t encode_normal_impl(uint32_t sign, int e, int m, int bias, float mant) {
    if (e > MaxExponent) {
        // 显式转换：从 uint32_t 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << SignShift) | static_cast<uint8_t>(InfValue));
    }
    if (e < SubnormalThreshold) {
        int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + SubnormalLdexpOffset)));
        if (sub <= 0) {
            // 显式转换：从 uint32_t 到 uint8_t（改变大小，但符号不变）
            return static_cast<uint8_t>(static_cast<uint8_t>(sign) << SignShift);
        }
        if (sub > MaxMantissa) {
            sub = MaxMantissa;
        }
        // 显式转换：从 int 到 uint8_t（改变符号），从 uint32_t 到 uint8_t
        return static_cast<uint8_t>((static_cast<uint8_t>(sign) << SignShift) | static_cast<uint8_t>(static_cast<unsigned int>(sub)));
    }
    // 显式转换：从 int 到 uint8_t（改变符号）
    uint8_t e_bits = static_cast<uint8_t>(static_cast<unsigned int>(e + bias));
    // 显式转换：从 uint32_t 到 uint8_t，从 int 到 uint8_t
    return static_cast<uint8_t>((static_cast<uint8_t>(sign) << SignShift) | static_cast<uint8_t>(e_bits << ExponentShift) | static_cast<uint8_t>(static_cast<unsigned int>(m) & static_cast<unsigned int>(MantissaMask)));
}

// 编码浮点数的通用模板函数
// 用于消除不同浮点类型类中重复的 encode_from_float 实现
template<int Bias, int SubnormalThreshold, int MantissaQuantization,
         typename EncodeSpecialFunc, typename EncodeSubnormalFunc, typename EncodeNormalFunc>
inline uint8_t encode_from_float_impl(float f,
                                      EncodeSpecialFunc encode_special,
                                      EncodeSubnormalFunc encode_subnormal,
                                      EncodeNormalFunc encode_normal) {
    Float32Components components = extract_float32_components(f);
    uint32_t sign = components.sign;
    uint32_t exp = components.exp;
    uint32_t frac = components.frac;

    uint8_t special = encode_special(sign, exp, frac);
    if (special != static_cast<uint8_t>(0)) {
        return special;
    }

    Float32Normalized normalized = normalize_float32_components(exp, frac);
    int e_unbiased = normalized.e_unbiased;
    float a = normalized.mant;

    constexpr int bias = Bias;
    if (exp == static_cast<uint32_t>(0) || e_unbiased < SubnormalThreshold) {
        return encode_subnormal(sign, a, e_unbiased);
    }

    // 正规数：mantissa in [1,2)
    int e = e_unbiased;
    float mant = a;
    // 量化尾数
    int m = rne_to_int(static_cast<double>(mant - 1.0f) * static_cast<double>(MantissaQuantization));
    if (m >= MantissaQuantization) {
        m = 0;
        ++e;
    }

    return encode_normal(sign, e, m, bias, mant);
}

}  // namespace dtypes
}  // namespace asnumpy

