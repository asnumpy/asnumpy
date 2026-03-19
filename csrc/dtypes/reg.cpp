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


#include <asnumpy/dtypes/acl_float_reg.hpp>
#include <asnumpy/dtypes/desc.hpp>
#include <asnumpy/dtypes/float_types.hpp>
#include <asnumpy/dtypes/np_import.hpp>

namespace asnumpy {
namespace dtypes {

// 显式特化所有 ACL 浮点类型的静态成员
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e5m2)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e4m3fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float8_e8m0)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(bfloat16)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float6_e2m3fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float6_e3m2fn)
EXPLICIT_INSTANTIATE_ACL_FLOAT_MANAGER(float4_e2m1fn)

// TypeDescriptor 具体类型的特化实现
// TypeDescriptor 对 float8_e5m2 的特化
template<>
struct TypeDescriptor<float8_e5m2> : ACLFloatManager<float8_e5m2> {
    using T = float8_e5m2;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e5m2";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e5m2";
    static constexpr const char* kTpDoc = "Float8 E5M2 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '5';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 float8_e4m3fn 的特化
template<>
struct TypeDescriptor<float8_e4m3fn> : ACLFloatManager<float8_e4m3fn> {
    using T = float8_e4m3fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e4m3fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e4m3fn";
    static constexpr const char* kTpDoc = "Float8 E4M3FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '6';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 float8_e8m0 的特化
template<>
struct TypeDescriptor<float8_e8m0> : ACLFloatManager<float8_e8m0> {
    using T = float8_e8m0;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float8_e8m0";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float8_e8m0";
    static constexpr const char* kTpDoc = "Float8 E8M0 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '7';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 bfloat16 的特化
template<>
struct TypeDescriptor<bfloat16> : ACLFloatManager<bfloat16> {
    using T = bfloat16;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "bfloat16";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.bfloat16";
    static constexpr const char* kTpDoc = "BFloat16 floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '8';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 float6_e2m3fn 的特化
template<>
struct TypeDescriptor<float6_e2m3fn> : ACLFloatManager<float6_e2m3fn> {
    using T = float6_e2m3fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float6_e2m3fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float6_e2m3fn";
    static constexpr const char* kTpDoc = "Float6 E2M3FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = '9';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 float6_e3m2fn 的特化
template<>
struct TypeDescriptor<float6_e3m2fn> : ACLFloatManager<float6_e3m2fn> {
    using T = float6_e3m2fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float6_e3m2fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float6_e3m2fn";
    static constexpr const char* kTpDoc = "Float6 E3M2FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = 'A';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// TypeDescriptor 对 float4_e2m1fn 的特化
template<>
struct TypeDescriptor<float4_e2m1fn> : ACLFloatManager<float4_e2m1fn> {
    using T = float4_e2m1fn;

    static constexpr bool is_floating = true;
    static constexpr bool is_integral = false;
    static constexpr bool is_complex = false;
    static constexpr const char* kTypeName = "float4_e2m1fn";
    static constexpr const char* kQualifiedTypeName = "asnumpy.dtypes.float4_e2m1fn";
    static constexpr const char* kTpDoc = "Float4 E2M1FN floating-point values";

    static constexpr char kNpyDescrKind = 'f';
    static constexpr char kNpyDescrType = 'B';
    static constexpr char kNpyDescrByteorder = '=';
    static constexpr int kSize = sizeof(T);
    static constexpr int kAlignment = alignof(T);
};

// 对外暴露统一初始化与注册入口
void InitAndRegisterDtypes() {
    // 1) 确保只导入一次 NumPy C API
    ImportNumpy();
    
    // 2) 直接注册所有 ACL 浮点类型
    FloatTypeRegistrar<float8_e5m2>::RegisterDtype();
    FloatTypeRegistrar<float8_e4m3fn>::RegisterDtype();
    FloatTypeRegistrar<float8_e8m0>::RegisterDtype();
    FloatTypeRegistrar<bfloat16>::RegisterDtype();
    FloatTypeRegistrar<float6_e2m3fn>::RegisterDtype();
    FloatTypeRegistrar<float6_e3m2fn>::RegisterDtype();
    FloatTypeRegistrar<float4_e2m1fn>::RegisterDtype();
}

// 检查所有类型是否已注册
bool AreAllACLFloatTypesRegistered() {
    return ACLFloatManager<float8_e5m2>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float8_e4m3fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float8_e8m0>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<bfloat16>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float6_e2m3fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float6_e3m2fn>::npy_type != NPY_NOTYPE &&
           ACLFloatManager<float4_e2m1fn>::npy_type != NPY_NOTYPE;
}

// 获取特定类型的 dtype 类型号
template<typename T>
int GetACLFloatTypeNum() {
    return ACLFloatManager<T>::npy_type;
}

// 获取特定类型的 dtype 描述符
template<typename T>
PyArray_Descr* GetACLFloatDescr() {
    if (ACLFloatManager<T>::npy_descr == nullptr) {
        return nullptr;
    }
    Py_INCREF(ACLFloatManager<T>::npy_descr);
    return ACLFloatManager<T>::npy_descr;
}

// 创建特定类型的数组
template<typename T>
PyObject* CreateACLFloatArray(const std::vector<npy_intp>& shape, const std::vector<float>& data = {}) {
    int type_num = GetACLFloatTypeNum<T>();
    if (type_num == NPY_NOTYPE) {
        PyErr_SetString(PyExc_RuntimeError, "dtype not registered");
        return nullptr;
    }
    
    // 创建数组
    PyObject* array = PyArray_EMPTY(static_cast<int>(shape.size()), 
                                  const_cast<npy_intp*>(shape.data()), 
                                  type_num, 0);
    if (array == nullptr) {
        return nullptr;
    }
    
    // 填充数据
    if (!data.empty()) {
        T* array_data = static_cast<T*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(array)));
        for (size_t i = 0; i < std::min(data.size(), static_cast<size_t>(PyArray_SIZE(reinterpret_cast<PyArrayObject*>(array)))); ++i) {
            array_data[i] = T(data[i]);
        }
    }
    
    return array;
}

// 从数组获取 float 数据
template<typename T>
std::vector<float> GetACLFloatArrayData(PyObject* array) {
    if (!PyArray_Check(array)) {
        return {};
    }
    
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(array);
    if (PyArray_TYPE(arr) != GetACLFloatTypeNum<T>()) {
        return {};
    }
    
    npy_intp size = PyArray_SIZE(arr);
    T* data = static_cast<T*>(PyArray_DATA(arr));
    
    std::vector<float> result;
    result.reserve(size);
    
    for (npy_intp i = 0; i < size; ++i) {
        result.push_back(static_cast<float>(data[i]));
    }
    
    return result;
}

// 获取类型对象指针（用于绑定）
template<typename T>
PyObject* GetACLFloatTypeObject() {
    return ACLFloatManager<T>::type_ptr;
}

}  // namespace dtypes
}  // namespace asnumpy