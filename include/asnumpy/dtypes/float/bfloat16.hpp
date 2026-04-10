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
#include <acl/acl.h>

namespace asnumpy {
namespace dtypes {

// 自定义 bfloat16 实现（移除 Eigen 依赖）
class bfloat16 {
private:
    uint16_t rep_;
    struct ConstructFromRepTag {};
    constexpr bfloat16(uint16_t rep, ConstructFromRepTag) : rep_(rep) {}

    static uint16_t encode_from_float(float f) {
        uint32_t u = bit_cast<uint32_t>(f);
        // bfloat16: 1符号位 + 8指数位 + 7尾数位
        // 直接截取 float32 的高16位
        return static_cast<uint16_t>(u >> 16);
    }

    static float decode_to_float(uint16_t bits) {
        // 将 bfloat16 位模式扩展到 float32
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        return bit_cast<float>(u);
    }

public:
    static constexpr int kBits = 16;
    static constexpr int kExponentBias = 127;
    static constexpr int kMantissaBits = 7;

    constexpr bfloat16() : rep_(0) {}
    explicit bfloat16(float f) : rep_(encode_from_float(f)) {}
    explicit bfloat16(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
    explicit bfloat16(int i) : rep_(encode_from_float(static_cast<float>(i))) {}

    constexpr uint16_t rep() const { return rep_; }

    static constexpr bfloat16 FromRep(uint16_t rep) {
        return bfloat16(rep, ConstructFromRepTag{});
    }

    explicit operator float() const { return decode_to_float(rep_); }
    explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
    explicit operator bool() const { return (rep_ & 0x7FFFu) != 0; }

    bfloat16 operator-() const { return FromRep(static_cast<uint16_t>(rep_ ^ 0x8000u)); }

    bfloat16 operator+(const bfloat16& other) const {
        return bfloat16(static_cast<float>(*this) + static_cast<float>(other));
    }
    bfloat16 operator-(const bfloat16& other) const {
        return bfloat16(static_cast<float>(*this) - static_cast<float>(other));
    }
    bfloat16 operator*(const bfloat16& other) const {
        return bfloat16(static_cast<float>(*this) * static_cast<float>(other));
    }
    bfloat16 operator/(const bfloat16& other) const {
        return bfloat16(static_cast<float>(*this) / static_cast<float>(other));
    }

    bool operator==(const bfloat16& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        return (a == b) || (std::isnan(a) && std::isnan(b));
    }
    bool operator!=(const bfloat16& other) const { return !(*this == other); }
    bool operator<(const bfloat16& other) const {
        float a = static_cast<float>(*this), b = static_cast<float>(other);
        if (std::isnan(a) || std::isnan(b)) return false;
        return a < b;
    }
    bool operator<=(const bfloat16& other) const { return *this < other || *this == other; }
    bool operator>(const bfloat16& other) const { return other < *this; }
    bool operator>=(const bfloat16& other) const { return other <= *this; }

    // ACL 枚举获取
    static constexpr aclDataType getACLenum() { return ACL_BF16; }
};

}  // namespace dtypes
}  // namespace asnumpy

