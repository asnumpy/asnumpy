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

#include <algorithm>
#if defined(__has_include)
#  if __has_include(<bit>)
#    include <bit>
#  endif
#endif
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <type_traits>

namespace asnumpy {
namespace dtypes {

// 安全 bit_cast（兼容 C++17；优先使用 std::bit_cast）
template <class To, class From>
inline To bit_cast(const From& src) noexcept {
    static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
    static_assert(std::is_trivially_copyable_v<From>, "bit_cast From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "bit_cast To must be trivially copyable");

#if defined(__cpp_lib_bit_cast) && (__cpp_lib_bit_cast >= 201806L)
    return std::bit_cast<To>(src);
#else
    To dst{};
    const auto* src_bytes = reinterpret_cast<const std::byte*>(&src);
    auto* dst_bytes = reinterpret_cast<std::byte*>(&dst);
    std::copy_n(src_bytes, sizeof(To), dst_bytes);
    return dst;
#endif
}

inline int rne_to_int(double x) {
    double f = std::floor(x);
    double frac = x - f;
    if (frac > 0.5) return static_cast<int>(f) + 1;
    if (frac < 0.5) return static_cast<int>(f);
    // ties to even
    return (static_cast<long long>(f) & 1LL) ? static_cast<int>(f) + 1
                                             : static_cast<int>(f);
}

}  // namespace dtypes
}  // namespace asnumpy

