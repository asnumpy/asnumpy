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
 
 
// 对外暴露统一初始化与注册入口
void InitAndRegisterDtypes() {
    // 1) 确保只导入一次 NumPy C API
    ImportNumpy();
    
    // 2) 直接注册所有 ACL 浮点类型
    FloatTypeRegistrar<float8_e5m2>::RegisterDtype();
}
 
// 检查所有类型是否已注册
bool AreAllACLFloatTypesRegistered() {
    return ACLFloatManager<float8_e5m2>::npy_type != NPY_NOTYPE;
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
 // 检查类型号是否是某个 ACL 浮点类型
 template<typename T>
 bool IsACLFloatType(int type_num) {
     int acl_float_type_num = GetACLFloatTypeNum<T>();
     return (type_num == acl_float_type_num && acl_float_type_num != NPY_NOTYPE);
 }
 
 // 显式实例化模板函数（用于链接）
template int GetACLFloatTypeNum<float8_e5m2>();
template PyObject* GetACLFloatTypeObject<float8_e5m2>();
template PyArray_Descr* GetACLFloatDescr<float8_e5m2>();
template bool IsACLFloatType<float8_e5m2>(int);
}  // namespace dtypes
}  // namespace asnumpy