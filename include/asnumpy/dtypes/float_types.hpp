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
 #include <type_traits>
 #include <cstring>
 #include <acl/acl.h>
 
 
 namespace asnumpy {
 namespace dtypes {
 
 // 使用 acl.h 中的枚举值，不再重复定义
 
 // 安全 bit_cast（兼容 C++17）
 template <class To, class From>
 inline To bit_cast(const From& src) {
     static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
     To dst;
     std::memcpy(&dst, &src, sizeof(To));
     return dst;
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
 
 class float8_e5m2 {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e5m2(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     static uint8_t encode_from_float(float f) {
         uint32_t u = bit_cast<uint32_t>(f);
         uint32_t sign = u >> 31;
         uint32_t exp = (u >> 23) & 0xFFu;
         uint32_t frac = u & 0x7FFFFFu;
 
         if (exp == 0xFFu) {
             // Inf/NaN
             if (frac == 0) {
                 return static_cast<uint8_t>((sign << 7) | 0b0'11111'00);
             }
             return static_cast<uint8_t>((sign << 7) | 0b0'11111'10);  // qNaN
         }
         if (exp == 0 && frac == 0) {
             // Preserve ±0
             return static_cast<uint8_t>(sign << 7);
         }
 
         int e_unbiased;
         float a;
         if (exp == 0) {
             // float32 次正规，规范化
             // a = (frac / 2^23) * 2^(1-127)
             e_unbiased = -126;
             a = std::ldexp(static_cast<float>(frac), -149);  // frac * 2^-149
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             a = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
         }
 
         // e5m2 参数
         constexpr int bias = 15;
 
         // 次正规到 e5m2：阈值 2^-14
         if (exp == 0 || e_unbiased < -14) {
             // 直接量化到 2^-16 网格
             float mag = (exp == 0) ? a : std::ldexp(a, e_unbiased);
             int m = rne_to_int(static_cast<double>(mag) * 65536.0);
             if (m <= 0) return static_cast<uint8_t>(sign << 7);
             if (m > 3) m = 3;
             return static_cast<uint8_t>((sign << 7) | static_cast<uint8_t>(m));
         }
 
         // 正规数：mantissa in [1,2)
         int e = e_unbiased;
         float mant = a;  // already in [1,2) when exp != 0
         if (exp == 0) {
             // shouldn't be here, handled above
         }
         // 量化到 2-bit 尾数 (步长 1/4)
         int m = rne_to_int(static_cast<double>(mant - 1.0f) * 4.0);
         if (m >= 4) { m = 0; ++e; }
 
         if (e > 15) {
             return static_cast<uint8_t>((sign << 7) | 0b0'11111'00);
         }
         if (e < -14) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 16)));
             if (sub <= 0) return static_cast<uint8_t>(sign << 7);
             if (sub > 3) sub = 3;
             return static_cast<uint8_t>((sign << 7) | static_cast<uint8_t>(sub));
         }
 
         uint8_t e_bits = static_cast<uint8_t>(e + bias);
         return static_cast<uint8_t>((sign << 7) | (e_bits << 2) | (m & 0x3));
     }
 
     static float decode_to_float(uint8_t bits) {
         uint8_t sign = (bits >> 7) & 0x1u;
         uint8_t exp = (bits >> 2) & 0x1Fu;
         uint8_t mant = bits & 0x3u;
         constexpr int bias = 15;
 
         if (exp == 0x1F) {
             if (mant == 0) {
                 return sign ? -std::numeric_limits<float>::infinity()
                             : std::numeric_limits<float>::infinity();
             }
             return std::numeric_limits<float>::quiet_NaN();
         }
         if (exp == 0) {
             float v = static_cast<float>(mant) * (1.0f / 65536.0f);  // 2^-16
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * 0.25f;
         int e = static_cast<int>(exp) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
     }
 
 public:
     static constexpr int kBits = 8;
     static constexpr int kExponentBias = 15;
     static constexpr int kMantissaBits = 2;
 
     constexpr float8_e5m2() : rep_(0) {}
 
     explicit float8_e5m2(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e5m2(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e5m2(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const { return rep_; }
 
     static constexpr float8_e5m2 FromRep(uint8_t rep) {
         return float8_e5m2(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { return decode_to_float(rep_); }
     explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
     explicit operator bool() const { return (rep_ & 0x7Fu) != 0; }
 
     float8_e5m2 operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x80u)); }
 
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
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         return (a == b) || (std::isnan(a) && std::isnan(b));
     }
     bool operator!=(const float8_e5m2& other) const { return !(*this == other); }
     bool operator<(const float8_e5m2& other) const {
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         if (std::isnan(a) || std::isnan(b)) return false;
         return a < b;
     }
     bool operator<=(const float8_e5m2& other) const { return *this < other || *this == other; }
     bool operator>(const float8_e5m2& other) const { return other < *this; }
     bool operator>=(const float8_e5m2& other) const { return other <= *this; }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() { return ACL_FLOAT8_E5M2; }
 };
 
 class float8_e4m3fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e4m3fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     static uint8_t encode_from_float(float f) {
         uint32_t u = bit_cast<uint32_t>(f);
         uint32_t sign = u >> 31;
         uint32_t exp = (u >> 23) & 0xFFu;
         uint32_t frac = u & 0x7FFFFFu;
 
         // 无 Inf（finite-only），Inf/NaN 统一编码为外层 NaN
         if (exp == 0xFFu) {
             return static_cast<uint8_t>((sign << 7) | 0b0'1111'111);
         }
         if (exp == 0 && frac == 0) {
             return static_cast<uint8_t>(sign << 7);  // 保留 ±0
         }
 
         int e_unbiased;
         float mant;
         if (exp == 0) {
             // float32 次正规：规范化到 (0,1)
             e_unbiased = -126;
             mant = std::ldexp(static_cast<float>(frac), -149);
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);  // [1,2)
         }
 
         constexpr int bias = 7;
         // e4m3fn 正规阈值：2^-6
         if (exp == 0 || e_unbiased < -6) {
             float mag = (exp == 0) ? mant : std::ldexp(mant, e_unbiased);
             int m = rne_to_int(static_cast<double>(mag) * 512.0);  // 2^9 网格
             if (m <= 0) return static_cast<uint8_t>(sign << 7);
             if (m > 7) m = 7;
             return static_cast<uint8_t>((sign << 7) | static_cast<uint8_t>(m));
         }
 
         int e = e_unbiased;
         int m = rne_to_int(static_cast<double>(mant - 1.0f) * 8.0);
         if (m >= 8) { m = 0; ++e; }
 
         // 最大指数 e_unbiased=8（exp_bits=0x0F）范围内保留外层 NaN
         if (e > 8) {
             return static_cast<uint8_t>((sign << 7) | 0b0'1111'111);
         }
         if (e < -6) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 9)));
             if (sub <= 0) return static_cast<uint8_t>(sign << 7);
             if (sub > 7) sub = 7;
             return static_cast<uint8_t>((sign << 7) | static_cast<uint8_t>(sub));
         }
 
         uint8_t e_bits = static_cast<uint8_t>(e + bias);
         if (e_bits == 0x0F && (m & 0x7) == 0x7) {
             return static_cast<uint8_t>((sign << 7) | 0b0'1111'111);
         }
         return static_cast<uint8_t>((sign << 7) | (e_bits << 3) | (m & 0x7));
     }
 
     static float decode_to_float(uint8_t bits) {
         uint8_t sign = (bits >> 7) & 0x1u;
         uint8_t exp = (bits >> 3) & 0x0Fu;
         uint8_t mant = bits & 0x7u;
         constexpr int bias = 7;
 
         // 外层 NaN
         if (exp == 0x0F && mant == 0x07) {
             return std::numeric_limits<float>::quiet_NaN();
         }
         if (exp == 0) {
             float v = static_cast<float>(mant) * (1.0f / 512.0f);  // 2^-9
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * (1.0f / 8.0f);
         int e = static_cast<int>(exp) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
     }
 
 public:
     static constexpr int kBits = 8;
     static constexpr int kExponentBias = 7;
     static constexpr int kMantissaBits = 3;
 
     constexpr float8_e4m3fn() : rep_(0) {}
 
     explicit float8_e4m3fn(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e4m3fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e4m3fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const { return rep_; }
 
     static constexpr float8_e4m3fn FromRep(uint8_t rep) {
         return float8_e4m3fn(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { return decode_to_float(rep_); }
     explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
     explicit operator bool() const { return (rep_ & 0x7Fu) != 0; }
 
     float8_e4m3fn operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x80u)); }
 
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
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         return (a == b) || (std::isnan(a) && std::isnan(b));
     }
     bool operator!=(const float8_e4m3fn& other) const { return !(*this == other); }
     bool operator<(const float8_e4m3fn& other) const {
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         if (std::isnan(a) || std::isnan(b)) return false;
         return a < b;
     }
     bool operator<=(const float8_e4m3fn& other) const { return *this < other || *this == other; }
     bool operator>(const float8_e4m3fn& other) const { return other < *this; }
     bool operator>=(const float8_e4m3fn& other) const { return other <= *this; }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() { return ACL_FLOAT8_E4M3FN; }
 };
 
 class float8_e8m0 {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float8_e8m0(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     static uint8_t encode_from_float(float f) {
         if (std::isnan(f)) return 0xFFu;
         if (f < 0.0f) return 0xFFu;  // 无符号：负值->NaN
         if (std::isinf(f)) return 0xFEu;  // 无 Inf，饱和到最大有限
         if (f == 0.0f) return 0x00u;      // 无 0 语义，映射到最小有限
 
         int e;
         float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5,1)
         // 最近 2^k：阈值为 sqrt(2)/2 ≈ 0.7071
         int k = (m < 0.7071067811865476f) ? (e - 1) : e;
         int code = k + 127;  // 偏置 127
         if (code < 0) code = 0;
         if (code > 254) code = 254;
         return static_cast<uint8_t>(code);
     }
 
     static float decode_to_float(uint8_t bits) {
         if (bits == 0xFFu) {
             return std::numeric_limits<float>::quiet_NaN();
         }
         int e = static_cast<int>(bits) - 127;
         return std::ldexp(1.0f, e);
     }
 
 public:
     static constexpr int kBits = 8;
     static constexpr int kExponentBias = 127;
     static constexpr int kMantissaBits = 0;
 
     constexpr float8_e8m0() : rep_(0) {}
 
     explicit float8_e8m0(float f) : rep_(encode_from_float(f)) {}
     explicit float8_e8m0(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float8_e8m0(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const { return rep_; }
 
     static constexpr float8_e8m0 FromRep(uint8_t rep) {
         return float8_e8m0(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { return decode_to_float(rep_); }
     explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
     explicit operator bool() const { return true; }  // 无 0 概念
 
     float8_e8m0 operator-() const { return FromRep(0xFFu); }  // 负号 -> NaN
 
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
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         return (a == b) || (std::isnan(a) && std::isnan(b));
     }
     bool operator!=(const float8_e8m0& other) const { return !(*this == other); }
     bool operator<(const float8_e8m0& other) const {
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         if (std::isnan(a) || std::isnan(b)) return false;
         return a < b;
     }
     bool operator<=(const float8_e8m0& other) const { return *this < other || *this == other; }
     bool operator>(const float8_e8m0& other) const { return other < *this; }
     bool operator>=(const float8_e8m0& other) const { return other <= *this; }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() { return ACL_FLOAT8_E8M0; }
 };
 
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
         if (exp == 0) {
             e_unbiased = -126;
             mant = std::ldexp(static_cast<float>(frac), -149);
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
         }
 
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
         if (exp == 0) {
             e_unbiased = -126;
             mant = std::ldexp(static_cast<float>(frac), -149);
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
         }
 
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
 
         if (exp == 0) {
             float v = static_cast<float>(mant) * (1.0f / 4.0f);  // 次正规
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * 0.25f;
         int e = static_cast<int>(exp) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
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
 
 class float4_e2m1fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float4_e2m1fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     static uint8_t encode_from_float(float f) {
         uint32_t u = bit_cast<uint32_t>(f);
         uint32_t sign = u >> 31;
         uint32_t exp = (u >> 23) & 0xFFu;
         uint32_t frac = u & 0x7FFFFFu;
 
         if (exp == 0xFFu) {
             return static_cast<uint8_t>((sign << 3) | 0b0'11'1);
         }
         if (exp == 0 && frac == 0) {
             return static_cast<uint8_t>(sign << 3);
         }
 
         int e_unbiased;
         float mant;
         if (exp == 0) {
             e_unbiased = -126;
             mant = std::ldexp(static_cast<float>(frac), -149);
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
         }
 
         constexpr int bias = 1;  // E2M1 bias
         // 正规阈值：2^0 = 1
         if (exp == 0 || e_unbiased < 0) {
             float mag = (exp == 0) ? mant : std::ldexp(mant, e_unbiased);
             int m = rne_to_int(static_cast<double>(mag) * 2.0);  // 1-bit mantissa
             if (m <= 0) return static_cast<uint8_t>(sign << 3);
             if (m > 1) m = 1;
             return static_cast<uint8_t>((sign << 3) | static_cast<uint8_t>(m));
         }
 
         int e = e_unbiased;
         int m = rne_to_int(static_cast<double>(mant - 1.0f) * 2.0);
         if (m >= 2) { m = 0; ++e; }
 
         if (e > 2) {  // 最大指数 0b11 -> e_unbiased=2
             return static_cast<uint8_t>((sign << 3) | 0b0'11'1);
         }
         if (e < 0) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 1)));
             if (sub <= 0) return static_cast<uint8_t>(sign << 3);
             if (sub > 1) sub = 1;
             return static_cast<uint8_t>((sign << 3) | static_cast<uint8_t>(sub));
         }
 
         uint8_t e_bits = static_cast<uint8_t>(e + bias);
         return static_cast<uint8_t>((sign << 3) | (e_bits << 1) | (m & 0x1));
     }
 
     static float decode_to_float(uint8_t bits) {
         uint8_t sign = (bits >> 3) & 0x1u;
         uint8_t exp = (bits >> 1) & 0x3u;  // 2-bit exponent
         uint8_t mant = bits & 0x1u;         // 1-bit mantissa
         constexpr int bias = 1;
 
         if (exp == 0) {
             float v = static_cast<float>(mant) * 0.5f;  // 次正规
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * 0.5f;
         int e = static_cast<int>(exp) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
     }
 
 public:
     static constexpr int kBits = 4;
     static constexpr int kExponentBias = 1;
     static constexpr int kMantissaBits = 1;
 
     constexpr float4_e2m1fn() : rep_(0) {}
 
     explicit float4_e2m1fn(float f) : rep_(encode_from_float(f)) {}
     explicit float4_e2m1fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float4_e2m1fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const { return rep_; }
 
     static constexpr float4_e2m1fn FromRep(uint8_t rep) {
         return float4_e2m1fn(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { return decode_to_float(rep_); }
     explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
     explicit operator bool() const { return (rep_ & 0x7u) != 0; }
 
     float4_e2m1fn operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x8u)); }
 
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
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         return (a == b) || (std::isnan(a) && std::isnan(b));
     }
     bool operator!=(const float4_e2m1fn& other) const { return !(*this == other); }
     bool operator<(const float4_e2m1fn& other) const {
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         if (std::isnan(a) || std::isnan(b)) return false;
         return a < b;
     }
     bool operator<=(const float4_e2m1fn& other) const { return *this < other || *this == other; }
     bool operator>(const float4_e2m1fn& other) const { return other < *this; }
     bool operator>=(const float4_e2m1fn& other) const { return other <= *this; }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() { return ACL_FLOAT4_E2M1; }
 };
 
 class float4_e1m2fn {
 private:
     uint8_t rep_;
     struct ConstructFromRepTag {};
     constexpr float4_e1m2fn(uint8_t rep, ConstructFromRepTag) : rep_(rep) {}
 
     static uint8_t encode_from_float(float f) {
         uint32_t u = bit_cast<uint32_t>(f);
         uint32_t sign = u >> 31;
         uint32_t exp = (u >> 23) & 0xFFu;
         uint32_t frac = u & 0x7FFFFFu;
 
         // MX 格式：无 Inf/NaN，饱和到最大有限值
         if (exp == 0xFFu) {
             return static_cast<uint8_t>((sign << 3) | 0b0'1'11);
         }
         if (exp == 0 && frac == 0) {
             return static_cast<uint8_t>(sign << 3);
         }
 
         int e_unbiased;
         float mant;
         if (exp == 0) {
             e_unbiased = -126;
             mant = std::ldexp(static_cast<float>(frac), -149);
         } else {
             e_unbiased = static_cast<int>(exp) - 127;
             mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
         }
 
         constexpr int bias = 0;  // E1M2 bias (2^(1-1)-1 = 0)
         // 正规阈值：2^0 = 1
         if (exp == 0 || e_unbiased < 0) {
             float mag = (exp == 0) ? mant : std::ldexp(mant, e_unbiased);
             int m = rne_to_int(static_cast<double>(mag) * 4.0);  // 2-bit mantissa
             if (m <= 0) return static_cast<uint8_t>(sign << 3);
             if (m > 3) m = 3;
             return static_cast<uint8_t>((sign << 3) | static_cast<uint8_t>(m));
         }
 
         int e = e_unbiased;
         int m = rne_to_int(static_cast<double>(mant - 1.0f) * 4.0);
         if (m >= 4) { m = 0; ++e; }
 
         if (e > 1) {  // 最大指数 0b1 -> e_unbiased=1
             return static_cast<uint8_t>((sign << 3) | 0b0'1'11);
         }
         if (e < 0) {
             int sub = rne_to_int(static_cast<double>(std::ldexp(mant, e + 2)));
             if (sub <= 0) return static_cast<uint8_t>(sign << 3);
             if (sub > 3) sub = 3;
             return static_cast<uint8_t>((sign << 3) | static_cast<uint8_t>(sub));
         }
 
         uint8_t e_bits = static_cast<uint8_t>(e + bias);
         return static_cast<uint8_t>((sign << 3) | (e_bits << 2) | (m & 0x3));
     }
 
     static float decode_to_float(uint8_t bits) {
         uint8_t sign = (bits >> 3) & 0x1u;
         uint8_t exp = (bits >> 2) & 0x1u;  // 1-bit exponent
         uint8_t mant = bits & 0x3u;         // 2-bit mantissa
         constexpr int bias = 0;
 
         if (exp == 0) {
             float v = static_cast<float>(mant) * (1.0f / 4.0f);  // 次正规
             return sign ? -v : v;
         }
         float base = 1.0f + static_cast<float>(mant) * 0.25f;
         int e = static_cast<int>(exp) - bias;
         float v = std::ldexp(base, e);
         return sign ? -v : v;
     }
 
 public:
     static constexpr int kBits = 4;
     static constexpr int kExponentBias = 0;
     static constexpr int kMantissaBits = 2;
 
     constexpr float4_e1m2fn() : rep_(0) {}
 
     explicit float4_e1m2fn(float f) : rep_(encode_from_float(f)) {}
     explicit float4_e1m2fn(double d) : rep_(encode_from_float(static_cast<float>(d))) {}
     explicit float4_e1m2fn(int i) : rep_(encode_from_float(static_cast<float>(i))) {}
 
     constexpr uint8_t rep() const { return rep_; }
 
     static constexpr float4_e1m2fn FromRep(uint8_t rep) {
         return float4_e1m2fn(rep, ConstructFromRepTag{});
     }
 
     explicit operator float() const { return decode_to_float(rep_); }
     explicit operator double() const { return static_cast<double>(static_cast<float>(*this)); }
     explicit operator bool() const { return (rep_ & 0x7u) != 0; }
 
     float4_e1m2fn operator-() const { return FromRep(static_cast<uint8_t>(rep_ ^ 0x8u)); }
 
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
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         return (a == b) || (std::isnan(a) && std::isnan(b));
     }
     bool operator!=(const float4_e1m2fn& other) const { return !(*this == other); }
     bool operator<(const float4_e1m2fn& other) const {
         float a = static_cast<float>(*this), b = static_cast<float>(other);
         if (std::isnan(a) || std::isnan(b)) return false;
         return a < b;
     }
     bool operator<=(const float4_e1m2fn& other) const { return *this < other || *this == other; }
     bool operator>(const float4_e1m2fn& other) const { return other < *this; }
     bool operator>=(const float4_e1m2fn& other) const { return other <= *this; }
 
     // ACL 枚举获取
     static constexpr aclDataType getACLenum() { return ACL_FLOAT4_E1M2; }
 };
 
 //  ACL 枚举获取：每个类型提供 getACLenum() 静态方法
 
 }  // namespace dtypes
 }  // namespace asnumpy
 
 