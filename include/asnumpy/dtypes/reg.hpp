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

#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>

namespace asnumpy {
namespace dtypes {

    // 对外暴露统一初始化与注册入口
    void InitAndRegisterDtypes();
    
    // 检查所有类型是否已注册
    bool AreAllACLFloatTypesRegistered();
    
    // 获取特定类型的 dtype 类型号
    template<typename T>
    int GetACLFloatTypeNum();
    
    // 获取特定类型的 dtype 描述符
    template<typename T>
    PyArray_Descr* GetACLFloatDescr();
    
    // 获取类型对象指针（用于绑定）
    template<typename T>
    PyObject* GetACLFloatTypeObject();
    
    // 创建特定类型的数组
    template<typename T>
    PyObject* CreateACLFloatArray(const std::vector<npy_intp>& shape, const std::vector<float>& data = {});
    
    // 从数组获取 float 数据
    template<typename T>
    std::vector<float> GetACLFloatArrayData(PyObject* array);

}  // namespace dtypes
}  // namespace asnumpy

