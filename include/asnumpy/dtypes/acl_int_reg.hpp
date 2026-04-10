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

#include <asnumpy/dtypes/np_import.hpp>
#include <asnumpy/dtypes/desc.hpp>
#include <asnumpy/dtypes/acl_manager_members.hpp>
#include <vector>
#include <algorithm>
#include <cstddef>

// NumPy 2.x 兼容：在 2.0 及以上使用 PyArray_DescrProto
#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace asnumpy {
namespace dtypes {

    // ACL 整数类型管理器基类
    template <typename T>
    struct ACLIntManager {
        ASNUMPY_ACL_MANAGER_COMMON_MEMBERS
        static int Dtype() { return npy_type; }
    };

    // Python 标量对象包装
    template <typename T>
    struct PyACLIntScalar {
        PyObject_HEAD;
        T value;
    };

    // 通用的整数类型注册器
    template <typename T>
    class IntTypeRegistrar {
    public:
        using ScalarType = PyACLIntScalar<T>;
        using DescriptorType = TypeDescriptor<T>;
        
        // 数组操作函数
        static void copyswap(void* dst, void* src, int swap, void* arr) {
            if (src != nullptr) {
                const auto* src_bytes = static_cast<const std::byte*>(src);
                auto* dst_bytes = static_cast<std::byte*>(dst);
                std::copy_n(src_bytes, sizeof(T), dst_bytes);
            }
            if (swap && sizeof(T) > 1) {
                // 对于多字节类型才需要字节序转换
                char* bytes = static_cast<char*>(dst);
                for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
            }
        }
        
        static void copyswapn(void* dst, npy_intp dstride, void* src, npy_intp sstride,
                            npy_intp n, int swap, void* arr) {
            char* dstptr = static_cast<char*>(dst);
            char* srcptr = static_cast<char*>(src);
            
            for (npy_intp i = 0; i < n; i++) {
                copyswap(dstptr, srcptr, swap, arr);
                dstptr += dstride;
                if (src != nullptr) {
                    srcptr += sstride;
                }
            }
        }
        
        static int compare(const void* a, const void* b, void* arr) {
            T ta = *static_cast<const T*>(a);
            T tb = *static_cast<const T*>(b);
            
            if (ta < tb) return -1;
            if (ta > tb) return 1;
            return 0;
        }
        
        static PyObject* getitem(void* data, void* arr) {
            T val = *static_cast<T*>(data);
            int64_t int_val = static_cast<int64_t>(val);
            return PyLong_FromLongLong(int_val);
        }
        
        static int setitem(PyObject* obj, void* data, void* arr) {
            int64_t val;
            if (PyLong_Check(obj)) {
                val = PyLong_AsLongLong(obj);
            } else if (PyFloat_Check(obj)) {
                val = static_cast<int64_t>(PyFloat_AsDouble(obj));
            } else {
                PyErr_SetString(PyExc_TypeError, "Cannot convert to int type");
                return -1;
            }
            
            *static_cast<T*>(data) = T(val);
            return 0;
        }
        
        // 类型转换函数
        static void cast_to_int32(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            T* from_ptr = static_cast<T*>(from);
            int32_t* to_ptr = static_cast<int32_t*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = static_cast<int32_t>(from_ptr[i]);
            }
        }
        
        static void cast_from_int32(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            int32_t* from_ptr = static_cast<int32_t*>(from);
            T* to_ptr = static_cast<T*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = T(from_ptr[i]);
            }
        }
        
        static void cast_to_int64(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            T* from_ptr = static_cast<T*>(from);
            int64_t* to_ptr = static_cast<int64_t*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = static_cast<int64_t>(from_ptr[i]);
            }
        }
        
        static void cast_from_int64(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            int64_t* from_ptr = static_cast<int64_t*>(from);
            T* to_ptr = static_cast<T*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = T(from_ptr[i]);
            }
        }
        
        // 添加与其他 ACL 整数类型的转换函数
        template<typename U>
        static void cast_to_acl_type(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            T* from_ptr = static_cast<T*>(from);
            U* to_ptr = static_cast<U*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = U(static_cast<int64_t>(from_ptr[i]));
            }
        }
        
        template<typename U>
        static void cast_from_acl_type(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
            U* from_ptr = static_cast<U*>(from);
            T* to_ptr = static_cast<T*>(to);
            
            for (npy_intp i = 0; i < n; i++) {
                to_ptr[i] = T(static_cast<int64_t>(from_ptr[i]));
            }
        }
        
        // Python 标量类型方法
        static void scalar_dealloc(ScalarType* self) {
            Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
        }
        
        static PyObject* scalar_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
            ScalarType* self = reinterpret_cast<ScalarType*>(type->tp_alloc(type, 0));
            if (self != nullptr) {
                self->value = T(0);
            }
            return reinterpret_cast<PyObject*>(self);
        }
        
        static int scalar_init(ScalarType* self, PyObject* args, PyObject* kwds) {
            PyObject* obj = nullptr;
            if (!PyArg_ParseTuple(args, "|O", &obj)) {
                return -1;
            }
            
            if (obj == nullptr) {
                self->value = T(0);
            } else if (PyLong_Check(obj)) {
                self->value = T(static_cast<int64_t>(PyLong_AsLongLong(obj)));
            } else if (PyFloat_Check(obj)) {
                self->value = T(static_cast<int64_t>(PyFloat_AsDouble(obj)));
            } else {
                PyErr_SetString(PyExc_TypeError, "Cannot convert to ACL int type");
                return -1;
            }
            
            return 0;
        }
        
        static PyObject* scalar_repr(ScalarType* self) {
            int64_t val = static_cast<int64_t>(self->value);
            return PyUnicode_FromFormat("%s(%lld)", TypeDescriptor<T>::kTypeName, val);
        }
        
        static PyObject* scalar_str(ScalarType* self) {
            int64_t val = static_cast<int64_t>(self->value);
            return PyUnicode_FromFormat("%lld", val);
        }
        
        static PyObject* scalar_richcompare(ScalarType* self, PyObject* other, int op) {
            if (!PyObject_TypeCheck(other, Py_TYPE(self))) {
                Py_RETURN_NOTIMPLEMENTED;
            }
            
            ScalarType* other_scalar = reinterpret_cast<ScalarType*>(other);
            T self_val = self->value;
            T other_val = other_scalar->value;
            
            bool result = false;
            switch (op) {
                case Py_LT:
                    result = self_val < other_val;
                    break;
                case Py_LE:
                    result = self_val <= other_val;
                    break;
                case Py_EQ:
                    result = self_val == other_val;
                    break;
                case Py_NE:
                    result = self_val != other_val;
                    break;
                case Py_GT:
                    result = self_val > other_val;
                    break;
                case Py_GE:
                    result = self_val >= other_val;
                    break;
                default:
                    Py_RETURN_NOTIMPLEMENTED;
            }
            
            return PyBool_FromLong(result ? 1 : 0);
        }
        
        // ACL 枚举获取方法
        static PyObject* scalar_getACLenum(ScalarType* self, PyObject* args) {
            // 调用C++类的静态方法获取ACL枚举值
            int acl_enum = static_cast<int>(T::getACLenum());
            return PyLong_FromLong(acl_enum);
        }
        
        // 获取 NumPy 数组函数
        static PyArray_ArrFuncs* GetArrFuncs() {
            static bool initialized = false;
            if (!initialized) {
                PyArray_InitArrFuncs(&ACLIntManager<T>::arr_funcs);
                ACLIntManager<T>::arr_funcs.copyswap = copyswap;
                ACLIntManager<T>::arr_funcs.copyswapn = copyswapn;
                ACLIntManager<T>::arr_funcs.compare = compare;
                ACLIntManager<T>::arr_funcs.getitem = getitem;
                ACLIntManager<T>::arr_funcs.setitem = setitem;
                initialized = true;
            }
            return &ACLIntManager<T>::arr_funcs;
        }

        // 构造 NumPy 描述符原型（兼容 NumPy 2.x）
        static PyArray_DescrProto GetDescrProto() {
            PyArray_DescrProto proto = {
                PyObject_HEAD_INIT(&PyArrayDescr_Type)
                nullptr, // typeobj - 稍后填写
                TypeDescriptor<T>::kNpyDescrKind,
                TypeDescriptor<T>::kNpyDescrType,
                TypeDescriptor<T>::kNpyDescrByteorder,
                NPY_USE_SETITEM,
                0, // type_num
                static_cast<int>(TypeDescriptor<T>::kSize),
                static_cast<int>(TypeDescriptor<T>::kAlignment),
                nullptr, // subarray
                nullptr, // fields
                nullptr, // names
                GetArrFuncs(),
                nullptr, // metadata
                nullptr, // c_metadata
                -1 // hash
            };
            return proto;
        }
        
        // 注册与其他 ACL 整数类型的转换函数
        static void RegisterACLTypeConversions(int type_num) {
            // 检查其他 ACL 类型是否已注册，如果已注册则注册转换函数
            RegisterConversionIfRegistered<int4>(type_num);
            RegisterConversionIfRegistered<uint1>(type_num);
        }
        
        template<typename U>
        static void RegisterConversionIfRegistered(int type_num) {
            if (ACLIntManager<U>::npy_type == NPY_NOTYPE) {
                return;
            }
            if (ACLIntManager<U>::npy_type == type_num) {
                return;
            }
            // 注册 T -> U 的转换
            PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), ACLIntManager<U>::npy_type,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_to_acl_type<U>));
            // 注册 U -> T 的转换
            PyArray_RegisterCastFunc(PyArray_DescrFromType(ACLIntManager<U>::npy_type), type_num,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_from_acl_type<U>));
        }
        
        // 创建 Python 类型对象
        static PyTypeObject* CreatePythonType() {
            static PyTypeObject* type_def = nullptr;
            static bool initialized = false;
            
            if (!initialized) {
                // 定义方法表
                static PyMethodDef scalar_methods[] = {
                    {"getACLenum", reinterpret_cast<PyCFunction>(scalar_getACLenum), METH_NOARGS, 
                     "Get ACL data type enumeration value"},
                    {nullptr, nullptr, 0, nullptr}
                };
                
                // 定义类型槽位
                static PyType_Slot type_slots[] = {
                    {Py_tp_new, reinterpret_cast<void*>(scalar_new)},
                    {Py_tp_repr, reinterpret_cast<void*>(scalar_repr)},
                    {Py_tp_str, reinterpret_cast<void*>(scalar_str)},
                    {Py_tp_doc, reinterpret_cast<void*>(const_cast<char*>(TypeDescriptor<T>::kTpDoc))},
                    {Py_tp_dealloc, reinterpret_cast<void*>(scalar_dealloc)},
                    {Py_tp_init, reinterpret_cast<void*>(scalar_init)},
                    {Py_tp_hash, reinterpret_cast<void*>(PyObject_HashNotImplemented)},
                    {Py_tp_richcompare, reinterpret_cast<void*>(scalar_richcompare)},
                    {Py_tp_methods, reinterpret_cast<void*>(scalar_methods)},
                    {0, nullptr},
                };

                // 定义类型规范
                static PyType_Spec type_spec = {
                    /*.name=*/TypeDescriptor<T>::kQualifiedTypeName,
                    /*.basicsize=*/static_cast<int>(sizeof(ScalarType)),
                    /*.itemsize=*/0,
                    /*.flags=*/Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                    /*.slots=*/type_slots,
                };

                // 继承自 NumPy 的通用标量类型
                PyObject* bases = PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type));
                if (bases == nullptr) {
                    return nullptr;
                }

                PyObject* type_obj = PyType_FromSpecWithBases(&type_spec, bases);
                Py_DECREF(bases);
                if (type_obj == nullptr) {
                    return nullptr;
                }

                type_def = reinterpret_cast<PyTypeObject*>(type_obj);
                initialized = true;
            }
            
            return type_def;
        }
        
        // 注册 dtype 到 NumPy
        static bool RegisterDtype() {
            // 1. 创建 Python 标量类型
            PyTypeObject* python_type = CreatePythonType();
            if (python_type == nullptr) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to create Python scalar type");
                return false;
            }
            
            // 2. 构造并注册 NumPy 描述符原型（NumPy 2.x）
            ACLIntManager<T>::npy_descr_proto = GetDescrProto();
            Py_SET_TYPE(&ACLIntManager<T>::npy_descr_proto, &PyArrayDescr_Type);
            ACLIntManager<T>::npy_descr_proto.typeobj = python_type;

            int type_num = PyArray_RegisterDataType(&ACLIntManager<T>::npy_descr_proto);
            if (type_num < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to register dtype");
                return false;
            }

            // 3. 保存类型信息
            ACLIntManager<T>::npy_type = type_num;
            ACLIntManager<T>::npy_descr = PyArray_DescrFromType(type_num);
            ACLIntManager<T>::type_ptr = reinterpret_cast<PyObject*>(python_type);

            // 4. 立即注册转换函数（在dtype注册后立即初始化，避免警告）
            RegisterConversionFunctions(type_num);
            
            return true;
        }
        
        // 注册转换函数的辅助方法
        static void RegisterConversionFunctions(int type_num) {
            // 注册与标准整数类型的转换（双向）
            PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_INT32,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_to_int32));
            PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_INT32), type_num,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_from_int32));

            PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), NPY_INT64,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_to_int64));
            PyArray_RegisterCastFunc(PyArray_DescrFromType(NPY_INT64), type_num,
                                     reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_from_int64));
            
            // 注册与其他 ACL 整数类型的转换（如果它们已注册）
            RegisterACLTypeConversions(type_num);
        }
    };

    // 静态成员定义模板（需要在使用的地方特化）
    template <typename T>
    PyObject* ACLIntManager<T>::type_ptr = nullptr;
    
    template <typename T>
    int ACLIntManager<T>::npy_type = NPY_NOTYPE;
    
    template <typename T>
    PyArray_ArrFuncs ACLIntManager<T>::arr_funcs = {};
    
    template <typename T>
    PyArray_DescrProto ACLIntManager<T>::npy_descr_proto = {};
    
    template <typename T>
    PyArray_Descr* ACLIntManager<T>::npy_descr = nullptr;


    // 显式特化宏，简化在具体实现中的代码
    #define EXPLICIT_INSTANTIATE_ACL_INT_MANAGER(T) \
        template<> \
        PyObject* ACLIntManager<T>::type_ptr = nullptr; \
        \
        template<> \
        int ACLIntManager<T>::npy_type = NPY_NOTYPE; \
        \
        template<> \
        PyArray_ArrFuncs ACLIntManager<T>::arr_funcs = {}; \
        \
        template<> \
        PyArray_Descr* ACLIntManager<T>::npy_descr = nullptr;

    // 获取类型对象指针的函数声明（用于绑定到 Python 模块）
    template<typename T>
    PyObject* GetACLIntTypeObject();

}
}

