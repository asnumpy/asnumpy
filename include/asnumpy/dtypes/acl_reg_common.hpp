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

#include <asnumpy/dtypes/np_import.hpp>
#include <asnumpy/dtypes/desc.hpp>
#include <algorithm>
#include <cstddef>
#include <type_traits>

// NumPy 2.x 兼容：在 2.0 及以上使用 PyArray_DescrProto
#if NPY_ABI_VERSION < 0x02000000
#define PyArray_DescrProto PyArray_Descr
#endif

namespace asnumpy {
namespace dtypes {

template <typename T>
struct ACLScalarBase;

template <typename T, template <typename> class Manager, template <typename> class Scalar, class Policy>
class ACLTypeRegistrar {
public:
    using ScalarType = Scalar<T>;
    using DescriptorType = TypeDescriptor<T>;

    static void copyswap(void* dst, void* src, int swap, void* arr) {
        if (src != nullptr) {
            const auto* src_bytes = static_cast<const std::byte*>(src);
            auto* dst_bytes = static_cast<std::byte*>(dst);
            std::copy_n(src_bytes, sizeof(T), dst_bytes);
        }
        if (swap && sizeof(T) > 1) {
            char* bytes = static_cast<char*>(dst);
            for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
            }
        }
    }

    static void copyswapn(void* dst, npy_intp dstride, void* src, npy_intp sstride, npy_intp n,
                          int swap, void* arr) {
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

    static PyObject* getitem(void* data, void* arr) { return Policy::template GetItem<T>(data); }
    static int setitem(PyObject* obj, void* data, void* arr) {
        return Policy::template SetItem<T>(obj, data);
    }

    template <typename Builtin>
    static void cast_to_builtin(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
        T* from_ptr = static_cast<T*>(from);
        Builtin* to_ptr = static_cast<Builtin*>(to);
        for (npy_intp i = 0; i < n; i++) {
            to_ptr[i] = static_cast<Builtin>(from_ptr[i]);
        }
    }

    template <typename Builtin>
    static void cast_from_builtin(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
        Builtin* from_ptr = static_cast<Builtin*>(from);
        T* to_ptr = static_cast<T*>(to);
        for (npy_intp i = 0; i < n; i++) {
            to_ptr[i] = T(from_ptr[i]);
        }
    }

    template <typename U>
    static void cast_to_acl_type(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
        T* from_ptr = static_cast<T*>(from);
        U* to_ptr = static_cast<U*>(to);
        for (npy_intp i = 0; i < n; i++) {
            to_ptr[i] = Policy::template ConvertToPeer<U>(from_ptr[i]);
        }
    }

    template <typename U>
    static void cast_from_acl_type(void* from, void* to, npy_intp n, void* fromarr, void* toarr) {
        U* from_ptr = static_cast<U*>(from);
        T* to_ptr = static_cast<T*>(to);
        for (npy_intp i = 0; i < n; i++) {
            to_ptr[i] = Policy::template ConvertFromPeer<U, T>(from_ptr[i]);
        }
    }

    static void scalar_dealloc(ScalarType* self) {
        Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
    }

    static PyObject* scalar_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
        ScalarType* self = reinterpret_cast<ScalarType*>(type->tp_alloc(type, 0));
        if (self != nullptr) {
            Policy::template InitZero<T>(self->value);
        }
        return reinterpret_cast<PyObject*>(self);
    }

    static int scalar_init(ScalarType* self, PyObject* args, PyObject* kwds) {
        PyObject* obj = nullptr;
        if (!PyArg_ParseTuple(args, "|O", &obj)) {
            return -1;
        }
        if (obj == nullptr) {
            Policy::template InitZero<T>(self->value);
            return 0;
        }
        return Policy::template ScalarInit<T>(obj, self->value);
    }

    static PyObject* scalar_repr(ScalarType* self) { return Policy::template ScalarRepr<T>(self); }
    static PyObject* scalar_str(ScalarType* self) { return Policy::template ScalarStr<T>(self); }

    static PyObject* scalar_richcompare(ScalarType* self, PyObject* other, int op) {
        if (!PyObject_TypeCheck(other, Py_TYPE(self))) {
            Py_RETURN_NOTIMPLEMENTED;
        }
        ScalarType* other_scalar = reinterpret_cast<ScalarType*>(other);
        T self_val = self->value;
        T other_val = other_scalar->value;

        bool result = false;
        switch (op) {
            case Py_LT: result = self_val < other_val; break;
            case Py_LE: result = self_val <= other_val; break;
            case Py_EQ: result = self_val == other_val; break;
            case Py_NE: result = self_val != other_val; break;
            case Py_GT: result = self_val > other_val; break;
            case Py_GE: result = self_val >= other_val; break;
            default: Py_RETURN_NOTIMPLEMENTED;
        }
        return PyBool_FromLong(result ? 1 : 0);
    }

    static PyObject* scalar_getACLenum(ScalarType* self, PyObject* args) {
        int acl_enum = static_cast<int>(T::getACLenum());
        return PyLong_FromLong(acl_enum);
    }

    static PyArray_ArrFuncs* GetArrFuncs() {
        static bool initialized = false;
        if (!initialized) {
            PyArray_InitArrFuncs(&Manager<T>::arr_funcs);
            Manager<T>::arr_funcs.copyswap = copyswap;
            Manager<T>::arr_funcs.copyswapn = copyswapn;
            Manager<T>::arr_funcs.compare = compare;
            Manager<T>::arr_funcs.getitem = getitem;
            Manager<T>::arr_funcs.setitem = setitem;
            initialized = true;
        }
        return &Manager<T>::arr_funcs;
    }

    static PyArray_DescrProto GetDescrProto() {
        PyArray_DescrProto proto = {
            PyObject_HEAD_INIT(&PyArrayDescr_Type) nullptr,
            DescriptorType::kNpyDescrKind,
            DescriptorType::kNpyDescrType,
            DescriptorType::kNpyDescrByteorder,
            NPY_USE_SETITEM,
            0,
            static_cast<int>(DescriptorType::kSize),
            static_cast<int>(DescriptorType::kAlignment),
            nullptr,
            nullptr,
            nullptr,
            GetArrFuncs(),
            nullptr,
            nullptr,
            -1};
        return proto;
    }

    template <typename U>
    static void RegisterConversionIfRegistered(int type_num) {
        if (Manager<U>::npy_type == NPY_NOTYPE) return;
        if (Manager<U>::npy_type == type_num) return;
        PyArray_RegisterCastFunc(PyArray_DescrFromType(type_num), Manager<U>::npy_type,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_to_acl_type<U>));
        PyArray_RegisterCastFunc(PyArray_DescrFromType(Manager<U>::npy_type), type_num,
                                 reinterpret_cast<PyArray_VectorUnaryFunc*>(cast_from_acl_type<U>));
    }

    static PyTypeObject* CreatePythonType() {
        static PyTypeObject* type_def = nullptr;
        static bool initialized = false;

        if (!initialized) {
            static PyMethodDef scalar_methods[] = {
                {"getACLenum", reinterpret_cast<PyCFunction>(scalar_getACLenum), METH_NOARGS,
                 "Get ACL data type enumeration value"},
                {nullptr, nullptr, 0, nullptr}};

            static PyType_Slot type_slots[] = {
                {Py_tp_new, reinterpret_cast<void*>(scalar_new)},
                {Py_tp_repr, reinterpret_cast<void*>(scalar_repr)},
                {Py_tp_str, reinterpret_cast<void*>(scalar_str)},
                {Py_tp_doc, reinterpret_cast<void*>(const_cast<char*>(DescriptorType::kTpDoc))},
                {Py_tp_dealloc, reinterpret_cast<void*>(scalar_dealloc)},
                {Py_tp_init, reinterpret_cast<void*>(scalar_init)},
                {Py_tp_hash, reinterpret_cast<void*>(PyObject_HashNotImplemented)},
                {Py_tp_richcompare, reinterpret_cast<void*>(scalar_richcompare)},
                {Py_tp_methods, reinterpret_cast<void*>(scalar_methods)},
                {0, nullptr},
            };

            static PyType_Spec type_spec = {
                DescriptorType::kQualifiedTypeName,
                static_cast<int>(sizeof(ScalarType)),
                0,
                Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
                type_slots,
            };

            PyObject* bases = PyTuple_Pack(1, reinterpret_cast<PyObject*>(&PyGenericArrType_Type));
            if (bases == nullptr) return nullptr;

            PyObject* type_obj = PyType_FromSpecWithBases(&type_spec, bases);
            Py_DECREF(bases);
            if (type_obj == nullptr) return nullptr;

            type_def = reinterpret_cast<PyTypeObject*>(type_obj);
            initialized = true;
        }

        return type_def;
    }

    static void RegisterConversionFunctions(int type_num) {
        Policy::template RegisterBuiltins<T, ACLTypeRegistrar>(type_num);
        Policy::template RegisterPeers<T, ACLTypeRegistrar>(type_num);
    }

    static bool RegisterDtype() {
        PyTypeObject* python_type = CreatePythonType();
        if (python_type == nullptr) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create Python scalar type");
            return false;
        }

        Manager<T>::npy_descr_proto = GetDescrProto();
        Py_SET_TYPE(&Manager<T>::npy_descr_proto, &PyArrayDescr_Type);
        Manager<T>::npy_descr_proto.typeobj = python_type;

        int type_num = PyArray_RegisterDataType(&Manager<T>::npy_descr_proto);
        if (type_num < 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to register dtype");
            return false;
        }

        Manager<T>::npy_type = type_num;
        Manager<T>::npy_descr = PyArray_DescrFromType(type_num);
        Manager<T>::type_ptr = reinterpret_cast<PyObject*>(python_type);

        RegisterConversionFunctions(type_num);
        return true;
    }
};

}  // namespace dtypes
}  // namespace asnumpy

