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


#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/math/trigonometric_functions.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_tan.h>
#include <aclnnop/aclnn_asin.h>
#include <aclnnop/aclnn_acos.h>
#include <aclnnop/aclnn_atan.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_atan2.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>

#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include <cmath>

namespace asnumpy {
    NPUArray Sin(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                           
            dtype,                                      
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnSinGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnSin(workspace, workspaceSize, executor, nullptr);
            },
            "Sin"                                       
        );
    }


    NPUArray Cos(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnCosGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnCos(workspace, workspaceSize, executor, nullptr);
            },
            "Cos"                                       
        );
    }


    NPUArray Tan(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || 
            x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnTanGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnTan(workspace, workspaceSize, executor, nullptr);
            },
            "Tan"                                       
        );
    }


    NPUArray Arcsin(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAsinGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnAsin(workspace, workspaceSize, executor, nullptr);
            },
            "Arcsin"                                       
        );
    }


    NPUArray Arccos(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAcosGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnAcos(workspace, workspaceSize, executor, nullptr);
            },
            "Arccos"                                       
        );
    }


    NPUArray Arctan(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
         py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteUnaryOp(
            x,                                          
            dtype,                                     
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAtanGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnAtan(workspace, workspaceSize, executor, nullptr);
            },
            "Arctan"                                       
        );
    }


    NPUArray Hypot(const NPUArray& a, const NPUArray& b) {
        auto broadcast = GetBroadcastShape(a, b);

        // 步骤1: 计算a的平方 (a²)
        NPUArray a_squared(a.shape, a.dtype);
        uint64_t a_sq_workspace_size = 0;
        aclOpExecutor* a_sq_executor = nullptr;
        auto error = aclnnMulGetWorkspaceSize(
            a.tensorPtr, a.tensorPtr,
            a_squared.tensorPtr,
            &a_sq_workspace_size,
            &a_sq_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        AclWorkspace a_sq_workspace(a_sq_workspace_size);

        error = aclnnMul(a_sq_workspace.get(), a_sq_workspace_size, a_sq_executor, nullptr);
        CheckAclnnStatus(error, "aclnnMul error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        // 步骤2: 计算b的平方 (b²)
        NPUArray b_squared(b.shape, b.dtype);
        uint64_t b_sq_workspace_size = 0;
        aclOpExecutor* b_sq_executor = nullptr;
        error = aclnnMulGetWorkspaceSize(
            b.tensorPtr, b.tensorPtr,
            b_squared.tensorPtr,
            &b_sq_workspace_size,
            &b_sq_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        AclWorkspace b_sq_workspace(b_sq_workspace_size);

        error = aclnnMul(b_sq_workspace.get(), b_sq_workspace_size, b_sq_executor, nullptr);
        CheckAclnnStatus(error, "aclnnMul error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        // 步骤3: 计算平方和 (a² + b²)
        auto dtype = a.aclDtype;
        if (a.aclDtype == b.aclDtype) {
            dtype = a.aclDtype;
        }
        else if (a.aclDtype == ACL_DOUBLE || b.aclDtype == ACL_DOUBLE){
            dtype = ACL_DOUBLE;
        }
        else if (a.aclDtype == ACL_FLOAT || b.aclDtype == ACL_FLOAT){
            dtype = ACL_FLOAT;
        }
        NPUArray sum_squares(broadcast, dtype);
        uint64_t add_workspace_size = 0;
        aclOpExecutor* add_executor = nullptr;
        aclScalar* alpha_scalar;
        if (dtype == ACL_DOUBLE || dtype == ACL_INT64){
            double alpha = 1.0;
            alpha_scalar = aclCreateScalar(&alpha, dtype);
        }
        else {
            float alpha = 1.0f;
            alpha_scalar = aclCreateScalar(&alpha, dtype);
        }
        
        error = aclnnAddGetWorkspaceSize(
            a_squared.tensorPtr, b_squared.tensorPtr,
            alpha_scalar, sum_squares.tensorPtr,
            &add_workspace_size, &add_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        AclWorkspace add_workspace(add_workspace_size);

        error = aclnnAdd(add_workspace.get(), add_workspace_size, add_executor, nullptr);
        CheckAclnnStatus(error, "aclnnAdd error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        // 步骤4: 计算平方根 (√(a² + b²))
        aclDataType aclType = ACL_DOUBLE;
        if (sum_squares.aclDtype == ACL_FLOAT || sum_squares.aclDtype == ACL_FLOAT16 || sum_squares.aclDtype == ACL_DOUBLE || sum_squares.aclDtype == ACL_COMPLEX64 || sum_squares.aclDtype == ACL_COMPLEX128){
            aclType = sum_squares.aclDtype;
        }
        NPUArray result(broadcast, aclType);
        uint64_t sqrt_workspace_size = 0;
        aclOpExecutor* sqrt_executor = nullptr;
        error = aclnnSqrtGetWorkspaceSize(
            sum_squares.tensorPtr, result.tensorPtr,
            &sqrt_workspace_size, &sqrt_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        AclWorkspace sqrt_workspace(sqrt_workspace_size);

        error = aclnnSqrt(sqrt_workspace.get(), sqrt_workspace_size, sqrt_executor, nullptr);
        CheckAclnnStatus(error, "aclnnSqrt error");

        // 同步设备并释放资源
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        aclDestroyScalar(alpha_scalar);

        return result;
    }


    NPUArray Arctan2(const NPUArray& y, const NPUArray& x) {
        auto aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_DOUBLE || y.aclDtype == ACL_DOUBLE){
            aclType = ACL_DOUBLE;
        }
        py::dtype dtype = NPUArray::GetPyDtype(aclType);
        return ExecuteBinaryOp(
            y,
            x,                                           
            dtype,                                     
            [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnAtan2GetWorkspaceSize(in1, in2, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnAtan2(workspace, workspaceSize, executor, nullptr);
            },
            "Arctan2"                                       
        );
    }


    NPUArray Radians(const NPUArray& x) {
        // 业务参数校验
        if (x.tensorSize == 0) {
            throw std::runtime_error("Radians: input tensor has no elements");
        }
        if (x.aclDtype != ACL_FLOAT && x.aclDtype != ACL_DOUBLE && x.aclDtype != ACL_FLOAT16) {
            throw std::runtime_error("Radians: input must be float, double or float16 type");
        }

        // 初始化结果张量（与输入同形状同类型）
        NPUArray result(x.shape, x.aclDtype);

        // 角度转弧度因子：π/180
        NPUArray scalar_factor({}, x.aclDtype);
        const double rad_factor = M_PI / 180.0;
        void* scalar_factor_ptr = nullptr;

        // 资源声明
        aclrtStream stream = nullptr;
        uint64_t workspace_size = 0;
        aclOpExecutor *executor = nullptr;
        void *workspace_addr = nullptr;
        aclTensorList *input_list = nullptr;  // 输入张量列表
        aclTensorList *output_list = nullptr; // 输出张量列表

        try {
            // 获取标量张量的设备指针
            auto error = aclGetRawTensorAddr(scalar_factor.tensorPtr, &scalar_factor_ptr);
            if (error != ACL_SUCCESS || !scalar_factor_ptr) {
                throw std::runtime_error(fmt::format("Failed to get scalar factor pointer, error: {}", error));
            }

            // 拷贝转换因子到设备（按数据类型适配）
            if (x.aclDtype == ACL_FLOAT) {
                float factor = static_cast<float>(rad_factor);
                aclrtMemcpy(scalar_factor_ptr, sizeof(float), &factor, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
            } else if (x.aclDtype == ACL_DOUBLE) {
                aclrtMemcpy(scalar_factor_ptr, sizeof(double), &rad_factor, sizeof(double), ACL_MEMCPY_HOST_TO_DEVICE);
            } else if (x.aclDtype == ACL_FLOAT16) {
                float factor_float = static_cast<float>(rad_factor);
                uint32_t float_bits;
                std::memcpy(&float_bits, &factor_float, sizeof(float_bits));
                uint16_t fp16_bits = static_cast<uint16_t>(
                    ((float_bits >> 16) & 0x8000U) |
                    (((float_bits >> 13) - 0x1C000U) & 0x7C00U) |
                    ((float_bits >> 13) & 0x03FFU)
                );
                aclrtMemcpy(scalar_factor_ptr, sizeof(uint16_t), &fp16_bits, sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
            }

            // 创建执行流
            aclrtCreateStream(&stream);

            // 关键修复：将单个张量包装为张量列表（匹配接口参数要求）
            // 输入张量列表（包含1个元素）
            aclTensor* input_tensors[] = {x.tensorPtr};
            input_list = aclCreateTensorList(input_tensors, 1);  // 直接用数组初始化列表
            
            // 输出张量列表（包含1个元素）
            aclTensor* output_tensors[] = {result.tensorPtr};
            output_list = aclCreateTensorList(output_tensors, 1);

            // 获取工作空间大小（使用张量列表作为参数）
            aclnnForeachMulScalarGetWorkspaceSize(
                input_list,          // 输入张量列表（第一个参数类型匹配）
                scalar_factor.tensorPtr,
                output_list,         // 输出张量列表
                &workspace_size,
                &executor
            );

            // 分配工作空间
            AclWorkspace workspace(workspace_size);

            // 执行标量乘法
            aclnnForeachMulScalar(
                workspace.get(),
                workspace_size,
                executor,
                stream
            );
            aclrtSynchronizeStream(stream);
        }
        catch (const std::exception& e) {
            // 释放资源（包含张量列表）
            if (input_list != nullptr) {
                aclDestroyTensorList(input_list);
            }
            if (output_list != nullptr) {
                aclDestroyTensorList(output_list);
            }
            if (stream != nullptr) {
                aclrtDestroyStream(stream);
            }
            throw;
        }

        return result;
    }

    NPUArray Degrees(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE) {
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);
        const double factor = 180.0 / M_PI;
        auto factorArr = NPUArray({1}, aclType);
        void* factorPtr = nullptr;
        auto error = aclGetRawTensorAddr(factorArr.tensorPtr, &factorPtr);
        CheckAclnnStatus(error, "Failed to get factor tensor pointer");
        double hostValue = factor;
        error = aclrtMemcpy(factorPtr, sizeof(double),
                            &hostValue, sizeof(double),
                            ACL_MEMCPY_HOST_TO_DEVICE);
        CheckAclnnStatus(error, "Write const factor error");
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        error = aclnnMulGetWorkspaceSize(
            x.tensorPtr,
            factorArr.tensorPtr,
            out.tensorPtr,
            &workspaceSize,
            &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);
        AclWorkspace workspace(workspaceSize);
        error = aclnnMul(workspace.get(), workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnMul error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return out;
    }
}
