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

namespace asnumpy {
namespace dtypes {

// 常量定义（避免魔法数字）
namespace constants {
    // 舍入相关常量
    constexpr double kRoundingThreshold = 0.5;  // 舍入到最近整数的阈值
    
    // float32 相关常量
    constexpr int kFloat32ExponentBias = 127;
    constexpr int kFloat32MantissaBits = 23;
    constexpr uint32_t kFloat32MantissaMask = 0x7FFFFFu;
    constexpr uint32_t kFloat32ExponentMask = 0xFFu;
    constexpr uint32_t kFloat32SignShift = 31;
    constexpr uint32_t kFloat32ExponentShift = 23;
    constexpr uint32_t kFloat32MaxExponent = 0xFFu;
    constexpr int kFloat32SubnormalExponent = -126;
    constexpr int kFloat32SubnormalLdexpOffset = -149;
    constexpr double kFloat32MantissaScale = 8388608.0;  // 2^23
    
    // float8_e5m2 相关常量
    constexpr int kFloat8E5M2Bias = 15;
    constexpr int kFloat8E5M2MantissaBits = 2;
    constexpr int kFloat8E5M2ExponentBits = 5;
    constexpr int kFloat8E5M2SubnormalThreshold = -14;
    constexpr double kFloat8E5M2SubnormalGrid = 65536.0;  // 2^16
    constexpr int kFloat8E5M2MaxMantissa = 3;  // 2^2 - 1
    constexpr int kFloat8E5M2MantissaQuantization = 4;  // 2^2
    constexpr int kFloat8E5M2MaxExponent = 15;
    constexpr int kFloat8E5M2SubnormalLdexpOffset = 16;
    constexpr uint8_t kFloat8E5M2SignShift = 7;
    constexpr uint8_t kFloat8E5M2ExponentShift = 2;
    constexpr uint8_t kFloat8E5M2MantissaMask = 0x3u;
    constexpr uint8_t kFloat8E5M2ExponentMask = 0x1Fu;
    constexpr uint8_t kFloat8E5M2MaxExponentValue = 0x1Fu;
    constexpr double kFloat8E5M2MantissaStep = 0.25;  // 1/4
    
    // float8_e4m3fn 相关常量
    constexpr int kFloat8E4M3FnBias = 7;
    constexpr int kFloat8E4M3FnMantissaBits = 3;
    constexpr int kFloat8E4M3FnExponentBits = 4;
    constexpr int kFloat8E4M3FnSubnormalThreshold = -6;
    constexpr double kFloat8E4M3FnSubnormalGrid = 512.0;  // 2^9
    constexpr int kFloat8E4M3FnMaxMantissa = 7;  // 2^3 - 1
    constexpr int kFloat8E4M3FnMantissaQuantization = 8;  // 2^3
    constexpr int kFloat8E4M3FnMaxExponent = 8;
    constexpr int kFloat8E4M3FnSubnormalLdexpOffset = 9;
    constexpr uint8_t kFloat8E4M3FnSignShift = 7;
    constexpr uint8_t kFloat8E4M3FnExponentShift = 3;
    constexpr uint8_t kFloat8E4M3FnMantissaMask = 0x7u;
    constexpr uint8_t kFloat8E4M3FnExponentMask = 0x0Fu;
    constexpr uint8_t kFloat8E4M3FnMaxExponentValue = 0x0Fu;
    constexpr double kFloat8E4M3FnMantissaStep = 0.125;  // 1/8
    
    // float8_e8m0 相关常量
    constexpr int kFloat8E8M0Bias = 127;
    constexpr int kFloat8E8M0MaxCode = 254;
    constexpr int kFloat8E8M0MinCode = 0;
    constexpr uint8_t kFloat8E8M0NaNValue = 0xFFu;
    constexpr uint8_t kFloat8E8M0InfValue = 0xFEu;
    constexpr uint8_t kFloat8E8M0ZeroValue = 0x00u;
    constexpr float kFloat8E8M0RoundingThreshold = 0.7071067811865476f;  // sqrt(2)/2
    
    // bfloat16 相关常量
    constexpr int kBfloat16ExponentBias = 127;
    constexpr int kBfloat16MantissaBits = 7;
    constexpr int kBfloat16BitShift = 16;
    
    // float6_e2m3fn 相关常量
    constexpr int kFloat6E2M3FnBias = 1;
    constexpr int kFloat6E2M3FnMantissaBits = 3;
    constexpr int kFloat6E2M3FnExponentBits = 2;
    constexpr int kFloat6E2M3FnSubnormalThreshold = 0;
    constexpr double kFloat6E2M3FnSubnormalGrid = 8.0;  // 2^3
    constexpr int kFloat6E2M3FnMaxMantissa = 7;  // 2^3 - 1
    constexpr int kFloat6E2M3FnMantissaQuantization = 8;  // 2^3
    constexpr int kFloat6E2M3FnMaxExponent = 2;
    constexpr int kFloat6E2M3FnSubnormalLdexpOffset = 3;
    constexpr uint8_t kFloat6E2M3FnSignShift = 5;
    constexpr uint8_t kFloat6E2M3FnExponentShift = 3;
    constexpr uint8_t kFloat6E2M3FnMantissaMask = 0x7u;
    constexpr uint8_t kFloat6E2M3FnExponentMask = 0x3u;
    constexpr double kFloat6E2M3FnMantissaStep = 0.125;  // 1/8
    
    // float6_e3m2fn 相关常量
    constexpr int kFloat6E3M2FnBias = 3;
    constexpr int kFloat6E3M2FnMantissaBits = 2;
    constexpr int kFloat6E3M2FnExponentBits = 3;
    constexpr int kFloat6E3M2FnSubnormalThreshold = -2;
    constexpr double kFloat6E3M2FnSubnormalGrid = 4.0;  // 2^2
    constexpr int kFloat6E3M2FnMaxMantissa = 3;  // 2^2 - 1
    constexpr int kFloat6E3M2FnMantissaQuantization = 4;  // 2^2
    constexpr int kFloat6E3M2FnMaxExponent = 4;
    constexpr int kFloat6E3M2FnSubnormalLdexpOffset = 2;
    constexpr uint8_t kFloat6E3M2FnSignShift = 5;
    constexpr uint8_t kFloat6E3M2FnExponentShift = 2;
    constexpr uint8_t kFloat6E3M2FnMantissaMask = 0x3u;
    constexpr uint8_t kFloat6E3M2FnExponentMask = 0x7u;
    constexpr double kFloat6E3M2FnMantissaStep = 0.25;  // 1/4
    
    // float4_e2m1fn 相关常量
    constexpr int kFloat4E2M1FnBias = 1;
    constexpr int kFloat4E2M1FnMantissaBits = 1;
    constexpr int kFloat4E2M1FnExponentBits = 2;
    constexpr int kFloat4E2M1FnSubnormalThreshold = 0;
    constexpr double kFloat4E2M1FnSubnormalGrid = 2.0;  // 2^1
    constexpr int kFloat4E2M1FnMaxMantissa = 1;  // 2^1 - 1
    constexpr int kFloat4E2M1FnMantissaQuantization = 2;  // 2^1
    constexpr int kFloat4E2M1FnMaxExponent = 2;
    constexpr int kFloat4E2M1FnSubnormalLdexpOffset = 1;
    constexpr uint8_t kFloat4E2M1FnSignShift = 3;
    constexpr uint8_t kFloat4E2M1FnExponentShift = 1;
    constexpr uint8_t kFloat4E2M1FnMantissaMask = 0x1u;
    constexpr uint8_t kFloat4E2M1FnExponentMask = 0x3u;
    constexpr double kFloat4E2M1FnMantissaStep = 0.5;  // 1/2
    
    // float4_e1m2fn 相关常量
    constexpr int kFloat4E1M2FnBias = 0;
    constexpr int kFloat4E1M2FnMantissaBits = 2;
    constexpr int kFloat4E1M2FnExponentBits = 1;
    constexpr int kFloat4E1M2FnSubnormalThreshold = 0;
    constexpr double kFloat4E1M2FnSubnormalGrid = 4.0;  // 2^2
    constexpr int kFloat4E1M2FnMaxMantissa = 3;  // 2^2 - 1
    constexpr int kFloat4E1M2FnMantissaQuantization = 4;  // 2^2
    constexpr int kFloat4E1M2FnMaxExponent = 1;
    constexpr int kFloat4E1M2FnSubnormalLdexpOffset = 2;
    constexpr uint8_t kFloat4E1M2FnSignShift = 3;
    constexpr uint8_t kFloat4E1M2FnExponentShift = 2;
    constexpr uint8_t kFloat4E1M2FnMantissaMask = 0x3u;
    constexpr uint8_t kFloat4E1M2FnExponentMask = 0x1u;
    constexpr double kFloat4E1M2FnMantissaStep = 0.25;  // 1/4
    
    // 位操作相关常量
    constexpr uint8_t kFloat8SignBitMask = 0x80u;  // 符号位掩码（第7位）
    constexpr uint8_t kFloat8MantissaMask7 = 0x7Fu;  // 7位尾数掩码（用于float8）
    constexpr uint8_t kFloat6SignBitMask = 0x20u;  // 符号位掩码（第5位，用于float6）
    constexpr uint8_t kFloat6MantissaMask5 = 0x1Fu;  // 5位尾数掩码（用于float6）
    constexpr uint8_t kFloat4SignBitMask = 0x8u;  // 符号位掩码（第3位，用于float4）
    constexpr uint8_t kFloat4MantissaMask3 = 0x7u;  // 3位尾数掩码（用于float4）
    constexpr uint16_t kBfloat16SignBitMask = 0x8000u;  // bfloat16符号位掩码（第15位）
    constexpr uint16_t kBfloat16MantissaMask15 = 0x7FFFu;  // 15位尾数掩码（用于bfloat16）
}  // namespace constants

}  // namespace dtypes
}  // namespace asnumpy

