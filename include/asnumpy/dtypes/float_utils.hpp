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

#include "float_constants.hpp"
#include <cstdint>
#include <cmath>
#include <type_traits>
#include <bit>

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

}  // namespace dtypes
}  // namespace asnumpy

