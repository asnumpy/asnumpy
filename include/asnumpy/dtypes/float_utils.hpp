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
#include <cstring>
#include <cstdlib>

namespace asnumpy {
namespace dtypes {

// 安全 bit_cast（兼容 C++17）
// 注意：使用 memcpy 是 C++17 之前实现 bit_cast 的标准方法
// 在 C++20 中可以使用 std::bit_cast，但为了兼容性保留此实现
 template <class To, class From>
 inline auto bit_cast(const From& src) -> To {
     static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
     static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");
     static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
     To dst;
     // 使用 memcpy 的返回值以满足 A0-1-2 规则
     void* const result = std::memcpy(&dst, &src, sizeof(To));
     // memcpy 在成功时返回目标指针，验证以确保正确性
     if (result != static_cast<void*>(&dst)) {
         std::abort();  // 不应该发生，但满足规则要求
     }
     return dst;
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

