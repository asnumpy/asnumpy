#pragma once

#include <acl/acl.h>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <sstream>

namespace asnumpy {
    namespace dtypes {

    // 亚字节整数模板基类
    // 将 N 位整数存储在一个字节（uint8_t）的低 N 位中，高位被忽略
    template <int N, typename UnderlyingTy>
    class SubByteInt {
    private:
        UnderlyingTy v_;
        using SignedUnderlyingTy = std::make_signed_t<UnderlyingTy>;
        using UnsignedUnderlyingTy = std::make_unsigned_t<UnderlyingTy>;
        static constexpr int kUnderlyingBits = std::numeric_limits<UnsignedUnderlyingTy>::digits;
    
        static_assert(
            std::is_same_v<UnderlyingTy, uint8_t> || std::is_same_v<UnderlyingTy, int8_t>,
            "底层类型必须是有符号或无符号 8 位整数");
    
        // 屏蔽高位
        static inline constexpr UnderlyingTy Mask(UnderlyingTy v) {
            return static_cast<UnsignedUnderlyingTy>(
                       static_cast<UnsignedUnderlyingTy>(v) << (kUnderlyingBits - N)) >>
                   (kUnderlyingBits - N);
        }
    
        // 屏蔽高位并对有符号类型进行符号扩展
        static inline constexpr UnderlyingTy ExtendToFullWidth(UnderlyingTy v) {
            return static_cast<UnderlyingTy>(static_cast<UnderlyingTy>(v)
                                             << (kUnderlyingBits - N)) >>
                   (kUnderlyingBits - N);
        }
    
        // 转换为对应的完整宽度 UnderlyingTy 值
        inline constexpr UnderlyingTy IntValue() const {
            return ExtendToFullWidth(v_);
        }
    
    public:
        using underlying_type = UnderlyingTy;
        static constexpr int bits = N;
        static constexpr int digits = std::is_signed_v<UnderlyingTy> ? N - 1 : N;
    
        constexpr SubByteInt() noexcept : v_(0) {}
        constexpr SubByteInt(const SubByteInt& other) noexcept = default;
        constexpr SubByteInt(SubByteInt&& other) noexcept = default;
        constexpr SubByteInt& operator=(const SubByteInt& other) = default;
        constexpr SubByteInt& operator=(SubByteInt&&) = default;
    
        explicit constexpr SubByteInt(UnderlyingTy val) : v_(Mask(val)) {}
        
        template <typename T>
        explicit constexpr SubByteInt(T t) : SubByteInt(static_cast<UnderlyingTy>(t)) {}
    
        static constexpr SubByteInt highest() { return SubByteInt((1 << digits) - 1); }
        static constexpr SubByteInt lowest() {
            return std::is_signed_v<UnderlyingTy> ? SubByteInt(1) << digits : SubByteInt(0);
        }
    
        // 类型转换
        template <typename T>
        explicit constexpr operator T() const {
            return static_cast<T>(IntValue());
        }
    
        // 取负
        constexpr SubByteInt operator-() const { return SubByteInt(-v_); }
    
        // 算术运算符
        constexpr SubByteInt operator+(const SubByteInt& other) const {
            return SubByteInt(v_ + other.v_);
        }
        constexpr SubByteInt operator-(const SubByteInt& other) const {
            return SubByteInt(v_ - other.v_);
        }
        constexpr SubByteInt operator*(const SubByteInt& other) const {
            return SubByteInt(v_ * other.v_);
        }
        constexpr SubByteInt operator/(const SubByteInt& other) const {
            return SubByteInt(IntValue() / other.IntValue());
        }
        constexpr SubByteInt operator%(const SubByteInt& other) const {
            return SubByteInt((IntValue() % other.IntValue()));
        }
    
        // 位运算符
        constexpr SubByteInt operator&(const SubByteInt& other) const {
            return SubByteInt(v_ & other.v_);
        }
        constexpr SubByteInt operator|(const SubByteInt& other) const {
            return SubByteInt(v_ | other.v_);
        }
        constexpr SubByteInt operator^(const SubByteInt& other) const {
            return SubByteInt(v_ ^ other.v_);
        }
        constexpr SubByteInt operator~() const { return SubByteInt(~v_); }
        constexpr SubByteInt operator>>(int amount) const {
            return SubByteInt(IntValue() >> amount);
        }
        constexpr SubByteInt operator<<(int amount) const { return SubByteInt(v_ << amount); }
    
        // 比较运算符
        constexpr bool operator==(const SubByteInt& other) const {
            return Mask(v_) == Mask(other.v_);
        }
        constexpr bool operator!=(const SubByteInt& other) const {
            return Mask(v_) != Mask(other.v_);
        }
        constexpr bool operator<(const SubByteInt& other) const {
            return IntValue() < other.IntValue();
        }
        constexpr bool operator>(const SubByteInt& other) const {
            return IntValue() > other.IntValue();
        }
        constexpr bool operator<=(const SubByteInt& other) const {
            return IntValue() <= other.IntValue();
        }
        constexpr bool operator>=(const SubByteInt& other) const {
            return IntValue() >= other.IntValue();
        }
    
        // 与 int64_t 的比较
        constexpr bool operator==(int64_t other) const { return IntValue() == other; }
        constexpr bool operator!=(int64_t other) const { return IntValue() != other; }
        constexpr bool operator<(int64_t other) const { return IntValue() < other; }
        constexpr bool operator>(int64_t other) const { return IntValue() > other; }
        constexpr bool operator<=(int64_t other) const { return IntValue() <= other; }
        constexpr bool operator>=(int64_t other) const { return IntValue() >= other; }
    
        friend constexpr bool operator==(int64_t a, const SubByteInt& b) {
            return a == b.IntValue();
        }
        friend constexpr bool operator!=(int64_t a, const SubByteInt& b) {
            return a != b.IntValue();
        }
        friend constexpr bool operator<(int64_t a, const SubByteInt& b) {
            return a < b.IntValue();
        }
        friend constexpr bool operator>(int64_t a, const SubByteInt& b) {
            return a > b.IntValue();
        }
        friend constexpr bool operator<=(int64_t a, const SubByteInt& b) {
            return a <= b.IntValue();
        }
        friend constexpr bool operator>=(int64_t a, const SubByteInt& b) {
            return a >= b.IntValue();
        }
    
        // 自增/自减
        constexpr SubByteInt& operator++() {
            v_ = Mask(v_ + 1);
            return *this;
        }
        constexpr SubByteInt operator++(int) {
            SubByteInt orig = *this;
            this->operator++();
            return orig;
        }
        constexpr SubByteInt& operator--() {
            v_ = Mask(v_ - 1);
            return *this;
        }
        constexpr SubByteInt operator--(int) {
            SubByteInt orig = *this;
            this->operator--();
            return orig;
        }
    
        // 复合赋值运算符
        constexpr SubByteInt& operator+=(const SubByteInt& other) {
            *this = *this + other;
            return *this;
        }
        constexpr SubByteInt& operator-=(const SubByteInt& other) {
            *this = *this - other;
            return *this;
        }
        constexpr SubByteInt& operator*=(const SubByteInt& other) {
            *this = *this * other;
            return *this;
        }
        constexpr SubByteInt& operator/=(const SubByteInt& other) {
            *this = *this / other;
            return *this;
        }
        constexpr SubByteInt& operator%=(const SubByteInt& other) {
            *this = *this % other;
            return *this;
        }
        constexpr SubByteInt& operator&=(const SubByteInt& other) {
            *this = *this & other;
            return *this;
        }
        constexpr SubByteInt& operator|=(const SubByteInt& other) {
            *this = *this | other;
            return *this;
        }
        constexpr SubByteInt& operator^=(const SubByteInt& other) {
            *this = *this ^ other;
            return *this;
        }
        constexpr SubByteInt& operator>>=(int amount) {
            *this = *this >> amount;
            return *this;
        }
        constexpr SubByteInt& operator<<=(int amount) {
            *this = *this << amount;
            return *this;
        }
    
        // 流输出
        friend std::ostream& operator<<(std::ostream& os, const SubByteInt& num) {
            os << static_cast<int16_t>(num);
            return os;
        }
    
        // 转字符串
        std::string ToString() const {
            std::ostringstream os;
            os << static_cast<int16_t>(*this);
            return os.str();
        }
    };
    
    // int4: 4位有符号整数，范围 [-8, 7]
    class int4 : public SubByteInt<4, int8_t> {
    private:
        using Base = SubByteInt<4, int8_t>;
    
    public:
        using Base::Base;
    
        // ACL 枚举获取
        static constexpr aclDataType getACLenum() { return ACL_INT4; }
    
        // 常量
        static constexpr int kBits = 4;
        static constexpr int kMin = -8;
        static constexpr int kMax = 7;
    };
    
    // uint1: 1位无符号整数，范围 [0, 1]
    class uint1 : public SubByteInt<1, uint8_t> {
    private:
        using Base = SubByteInt<1, uint8_t>;
    
    public:
        using Base::Base;
    
        // ACL 枚举获取
        static constexpr aclDataType getACLenum() { return ACL_UINT1; }
    
        // 常量
        static constexpr int kBits = 1;
        static constexpr int kMin = 0;
        static constexpr int kMax = 1;
    };
    
    }  // namespace dtypes
}  // namespace asnumpy

// std::numeric_limits 特化
namespace std {

    template <>
    struct numeric_limits<asnumpy::dtypes::int4> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = true;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NaN = false;
        static constexpr bool has_signaling_NaN = false;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr float_round_style round_style = round_toward_zero;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = false;
        static constexpr int digits = 3;  // N - 1 for signed
        static constexpr int digits10 = 0;  // floor(3 * log10(2)) = 0
        static constexpr int max_digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = 0;
        static constexpr int min_exponent10 = 0;
        static constexpr int max_exponent = 0;
        static constexpr int max_exponent10 = 0;
        static constexpr bool traps = true;
        static constexpr bool tinyness_before = false;
    
        static constexpr asnumpy::dtypes::int4 min() noexcept { 
            return asnumpy::dtypes::int4(-8); 
        }
        static constexpr asnumpy::dtypes::int4 lowest() noexcept { 
            return asnumpy::dtypes::int4(-8); 
        }
        static constexpr asnumpy::dtypes::int4 max() noexcept { 
            return asnumpy::dtypes::int4(7); 
        }
        static constexpr asnumpy::dtypes::int4 epsilon() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
        static constexpr asnumpy::dtypes::int4 round_error() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
        static constexpr asnumpy::dtypes::int4 infinity() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
        static constexpr asnumpy::dtypes::int4 quiet_NaN() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
        static constexpr asnumpy::dtypes::int4 signaling_NaN() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
        static constexpr asnumpy::dtypes::int4 denorm_min() noexcept { 
            return asnumpy::dtypes::int4(0); 
        }
    };
    
    template <>
    struct numeric_limits<asnumpy::dtypes::uint1> {
        static constexpr bool is_specialized = true;
        static constexpr bool is_signed = false;
        static constexpr bool is_integer = true;
        static constexpr bool is_exact = true;
        static constexpr bool has_infinity = false;
        static constexpr bool has_quiet_NaN = false;
        static constexpr bool has_signaling_NaN = false;
        static constexpr float_denorm_style has_denorm = denorm_absent;
        static constexpr bool has_denorm_loss = false;
        static constexpr float_round_style round_style = round_toward_zero;
        static constexpr bool is_iec559 = false;
        static constexpr bool is_bounded = true;
        static constexpr bool is_modulo = true;  // unsigned is modulo
        static constexpr int digits = 1;  // N for unsigned
        static constexpr int digits10 = 0;  // floor(1 * log10(2)) = 0
        static constexpr int max_digits10 = 0;
        static constexpr int radix = 2;
        static constexpr int min_exponent = 0;
        static constexpr int min_exponent10 = 0;
        static constexpr int max_exponent = 0;
        static constexpr int max_exponent10 = 0;
        static constexpr bool traps = true;
        static constexpr bool tinyness_before = false;
    
        static constexpr asnumpy::dtypes::uint1 min() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 lowest() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 max() noexcept { 
            return asnumpy::dtypes::uint1(1); 
        }
        static constexpr asnumpy::dtypes::uint1 epsilon() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 round_error() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 infinity() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 quiet_NaN() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 signaling_NaN() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
        static constexpr asnumpy::dtypes::uint1 denorm_min() noexcept { 
            return asnumpy::dtypes::uint1(0); 
        }
    };
    
    }  // namespace std