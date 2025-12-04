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


#include "asnumpy/utils/npu_scalar.hpp"
#include <fmt/core.h>
#include <cstring>

/*
    Helper function to safely cast values to target types.
    For float8_e5m2, converts to float first, then to target type.
    For other types, performs direct cast.
*/
template<typename TargetType, typename SourceType>
auto safe_cast(SourceType value) -> TargetType {
    if constexpr (std::is_same_v<std::decay_t<SourceType>, asnumpy::dtypes::float8_e5m2>) {
        return static_cast<TargetType>(static_cast<float>(value));
    } else {
        return static_cast<TargetType>(value);
    }
}

/*
    Creates an aclScalar object by automatically determining the appropriate ACL data type
    based on the C++ type of the input value. Uses TypeToACLDtype for compile-time mapping.
*/
template <typename T>
aclScalar* CreateScalar(T value) {
    return CreateScalar(value, TypeToACLDtype<std::decay_t<T>>::value);
}


// 辅助函数：处理类型匹配的标量创建
template <typename ValueType, typename TargetType, aclDataType DType>
aclScalar* create_scalar_typed(ValueType value) {
    if constexpr (std::is_same_v<std::decay_t<ValueType>, TargetType>) {
        return aclCreateScalar(&value, DType);
    } else {
        auto converted = safe_cast<TargetType>(value);
        return aclCreateScalar(&converted, DType);
    }
}

// 辅助函数：处理 float8_e5m2 类型
template <typename ValueType>
aclScalar* create_scalar_float8_e5m2(ValueType value) {
    uint8_t converted;
    if constexpr (std::is_same_v<std::decay_t<ValueType>, asnumpy::dtypes::float8_e5m2>) {
        converted = value.rep();
    } else {
        float f_value = static_cast<float>(value);
        asnumpy::dtypes::float8_e5m2 f8_value(f_value);
        converted = f8_value.rep();
    }
    
    aclScalar* result = aclCreateScalar(&converted, ACL_FLOAT8_E5M2);
    if (result == nullptr) {
        const char* error_msg = aclGetRecentErrMsg();
        std::string full_error_msg = fmt::format(
            "aclCreateScalar failed for ACL_FLOAT8_E5M2: "
            "converted value: 0x{:02x} ({})",
            converted, converted
        );
        
        if (error_msg != nullptr && std::strlen(error_msg) > static_cast<size_t>(0)) {
            full_error_msg += fmt::format(". Details: {}", error_msg);
        } else {
            full_error_msg += ". ACL may not support scalar creation for this type.";
        }
        
        throw std::runtime_error(full_error_msg);
    }
    return result;
}

// 辅助函数：处理转换为 uint16_t 的浮点类型
template <typename ValueType>
aclScalar* create_scalar_float16_like(ValueType value, aclDataType dtype) {
    auto converted = static_cast<uint16_t>(static_cast<float>(value));
    return aclCreateScalar(&converted, dtype);
}

// 辅助函数：处理转换为 uint8_t 的浮点类型
template <typename ValueType>
aclScalar* create_scalar_float8_like(ValueType value, aclDataType dtype) {
    auto converted = static_cast<uint8_t>(static_cast<float>(value));
    return aclCreateScalar(&converted, dtype);
}

// 辅助函数：处理复数类型
template <typename ValueType>
aclScalar* create_scalar_complex(ValueType value, aclDataType dtype) {
    if (dtype == ACL_COMPLEX128) {
        auto converted = std::complex<double>(static_cast<double>(value), 0.0);
        return aclCreateScalar(&converted, dtype);
    } else {
        auto converted = std::complex<float>(static_cast<float>(value), 0.0f);
        return aclCreateScalar(&converted, dtype);
    }
}

// 辅助函数：处理字符串类型
template <typename ValueType>
aclScalar* create_scalar_string(ValueType value) {
    double d_value = safe_cast<double>(value);
    std::string str_value = std::to_string(d_value);
    return aclCreateScalar(const_cast<char*>(str_value.c_str()), ACL_STRING);
}

// 辅助函数：处理未定义类型
template <typename ValueType>
aclScalar* create_scalar_undefined(ValueType /*value*/) {
    auto converted = static_cast<int32_t>(0);
    return aclCreateScalar(&converted, ACL_INT32);
}

/*
    Creates an aclScalar object for a given value with explicit data type control.
    Performs optimized value conversion when the input type matches the target data type.
    Falls back to static_cast conversion when types differ.
*/
template <typename ValueType>
aclScalar* CreateScalar(ValueType value, aclDataType dtype) {
    switch (dtype) {
        case ACL_FLOAT:
            return create_scalar_typed<ValueType, float, ACL_FLOAT>(value);
        case ACL_DOUBLE:
            return create_scalar_typed<ValueType, double, ACL_DOUBLE>(value);
        case ACL_INT32:
            return create_scalar_typed<ValueType, int32_t, ACL_INT32>(value);
        case ACL_INT64:
            return create_scalar_typed<ValueType, int64_t, ACL_INT64>(value);
        case ACL_INT8:
            return create_scalar_typed<ValueType, int8_t, ACL_INT8>(value);
        case ACL_INT16:
            return create_scalar_typed<ValueType, int16_t, ACL_INT16>(value);
        case ACL_UINT8:
            return create_scalar_typed<ValueType, uint8_t, ACL_UINT8>(value);
        case ACL_UINT16:
            return create_scalar_typed<ValueType, uint16_t, ACL_UINT16>(value);
        case ACL_UINT32:
            return create_scalar_typed<ValueType, uint32_t, ACL_UINT32>(value);
        case ACL_UINT64:
            return create_scalar_typed<ValueType, uint64_t, ACL_UINT64>(value);
        case ACL_BOOL:
            return create_scalar_typed<ValueType, bool, ACL_BOOL>(value);
        case ACL_FLOAT16:
        case ACL_BF16:
            return create_scalar_float16_like(value, dtype);
        case ACL_INT4:
            return create_scalar_typed<ValueType, int8_t, ACL_INT4>(value);
        case ACL_UINT1:
            return create_scalar_typed<ValueType, uint8_t, ACL_UINT1>(value);
        case ACL_COMPLEX64:
        case ACL_COMPLEX128:
        case ACL_COMPLEX32:
            return create_scalar_complex(value, dtype);
        case ACL_HIFLOAT8:
        case ACL_FLOAT8_E4M3FN:
        case ACL_FLOAT8_E8M0:
        case ACL_FLOAT6_E3M2:
        case ACL_FLOAT6_E2M3:
        case ACL_FLOAT4_E2M1:
        case ACL_FLOAT4_E1M2:
            return create_scalar_float8_like(value, dtype);
        case ACL_FLOAT8_E5M2:
            return create_scalar_float8_e5m2(value);
        case ACL_STRING:
            return create_scalar_string(value);
        case ACL_DT_UNDEFINED:
            return create_scalar_undefined(value);
        default:
            throw std::runtime_error("Unsupported dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}

// =====================
// Explicit instantiations
// =====================

// 1) CreateScalar(T value) —— 自动类型推导版本
template aclScalar* CreateScalar<float>(float);
template aclScalar* CreateScalar<double>(double);
template aclScalar* CreateScalar<int32_t>(int32_t);
template aclScalar* CreateScalar<int64_t>(int64_t);
template aclScalar* CreateScalar<int16_t>(int16_t);
template aclScalar* CreateScalar<int8_t>(int8_t);
template aclScalar* CreateScalar<uint64_t>(uint64_t);
template aclScalar* CreateScalar<uint32_t>(uint32_t);
template aclScalar* CreateScalar<uint16_t>(uint16_t);
template aclScalar* CreateScalar<uint8_t>(uint8_t);
template aclScalar* CreateScalar<bool>(bool);
template aclScalar* CreateScalar<asnumpy::dtypes::float8_e5m2>(asnumpy::dtypes::float8_e5m2);

// 2) CreateScalar(ValueType value, aclDataType dtype) —— 显式 dtype 版本
template aclScalar* CreateScalar<float>(float, aclDataType);
template aclScalar* CreateScalar<double>(double, aclDataType);
template aclScalar* CreateScalar<int32_t>(int32_t, aclDataType);
template aclScalar* CreateScalar<int64_t>(int64_t, aclDataType);
template aclScalar* CreateScalar<int16_t>(int16_t, aclDataType);
template aclScalar* CreateScalar<int8_t>(int8_t, aclDataType);
template aclScalar* CreateScalar<uint64_t>(uint64_t, aclDataType);
template aclScalar* CreateScalar<uint32_t>(uint32_t, aclDataType);
template aclScalar* CreateScalar<uint16_t>(uint16_t, aclDataType);
template aclScalar* CreateScalar<uint8_t>(uint8_t, aclDataType);
template aclScalar* CreateScalar<bool>(bool, aclDataType);
template aclScalar* CreateScalar<asnumpy::dtypes::float8_e5m2>(asnumpy::dtypes::float8_e5m2, aclDataType);