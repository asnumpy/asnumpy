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

// Prevent multiple inclusion
#pragma once

// 前向声明，避免循环依赖
namespace asnumpy {
namespace dtypes {

    // 前向声明 ACL 浮点类型
    class float8_e5m2;
    class float8_e4m3fn;
    class float8_e8m0;
    class bfloat16;
    class float6_e2m3fn;
    class float6_e3m2fn;
    class float4_e2m1fn;
    class float4_e1m2fn;
    
    // 前向声明 ACL 整数类型
    class int4;
    class uint1;

    // 前向声明 ACLFloatManager 和 ACLIntManager
    template <typename T>
    struct ACLFloatManager;
    
    template <typename T>
    struct ACLIntManager;

    // TypeDescriptor 模板声明
    template <typename T, typename Enable = void>
    struct TypeDescriptor {
    };

    // 具体类型的特化声明（实现在 reg.cpp 中）
    template<>
    struct TypeDescriptor<float8_e5m2>;

    template<>
    struct TypeDescriptor<float8_e4m3fn>;

    template<>
    struct TypeDescriptor<float8_e8m0>;

    template<>
    struct TypeDescriptor<bfloat16>;

    template<>
    struct TypeDescriptor<float6_e2m3fn>;

    template<>
    struct TypeDescriptor<float6_e3m2fn>;

    template<>
    struct TypeDescriptor<float4_e2m1fn>;

    template<>
    struct TypeDescriptor<float4_e1m2fn>;

    template<>
    struct TypeDescriptor<int4>;

    template<>
    struct TypeDescriptor<uint1>;

}
}
