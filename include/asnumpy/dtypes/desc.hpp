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

// forward declarations to avoid circular dependency
namespace asnumpy {
namespace dtypes {

// forward declarations for ACL float types
class float8_e5m2;
class float8_e4m3fn;
class float8_e8m0;
class bfloat16;
class float6_e2m3fn;
class float6_e3m2fn;
class float4_e2m1fn;
class float4_e1m2fn;

// forward declarations for ACL integer types
class int4;
class uint1;

// forward declarations for ACLFloatManager and ACLIntManager
template <typename T> struct ACLFloatManager;

template <typename T> struct ACLIntManager;

// TypeDescriptor template declaration
template <typename T, typename Enable = void> struct TypeDescriptor {};

// concrete type specialization declarations (implemented in reg.cpp)
template <> struct TypeDescriptor<float8_e5m2>;

template <> struct TypeDescriptor<float8_e4m3fn>;

template <> struct TypeDescriptor<float8_e8m0>;

template <> struct TypeDescriptor<bfloat16>;

template <> struct TypeDescriptor<float6_e2m3fn>;

template <> struct TypeDescriptor<float6_e3m2fn>;

template <> struct TypeDescriptor<float4_e2m1fn>;

template <> struct TypeDescriptor<float4_e1m2fn>;

template <> struct TypeDescriptor<int4>;

template <> struct TypeDescriptor<uint1>;

} // namespace dtypes
} // namespace asnumpy
