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
#include <fmt/base.h>
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
            x,                                           // input array
            dtype,                                       // output dtype
            [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnSinGetWorkspaceSize(in, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnSin(workspace, workspaceSize, executor, nullptr);
            },
            "Sin"                                        // operator name for logging
        );
    }


    NPUArray Cos(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnCosGetWorkspaceSize(
            x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnCos(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnCos error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }

        return out;
    }


    NPUArray Tan(const NPUArray& x) {
        aclDataType aclType = ACL_DOUBLE;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE || x.aclDtype == ACL_COMPLEX64 || x.aclDtype == ACL_COMPLEX128){
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnTanGetWorkspaceSize(
            x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnTan(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnTan error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }

        return out;
    }


    NPUArray Arcsin(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnAsinGetWorkspaceSize(
            x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnAsin(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnAsin error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }

        return out;
    }


    NPUArray Arccos(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnAcosGetWorkspaceSize(
            x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnAcos(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnAcos error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }

        return out;
    }


    NPUArray Arctan(const NPUArray& x) {
        aclDataType aclType = ACL_FLOAT;
        if (x.aclDtype == ACL_FLOAT || x.aclDtype == ACL_FLOAT16 || x.aclDtype == ACL_DOUBLE){
            aclType = x.aclDtype;
        }
        auto out = NPUArray(x.shape, aclType);

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor;

        auto error = aclnnAtanGetWorkspaceSize(x.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);

        void *workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if(error != ACL_SUCCESS) {
                throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtMalloc error = {}", error));
            }
        }

        error = aclnnAtan(workspaceAddr, workspaceSize, executor, nullptr);
        if(error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[math.cpp](arctan) aclnnAtan error = {}", error));
        }

        error = aclrtSynchronizeDevice();
        if(error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("[math.cpp](arctan) aclrtSynchronizeDevice error = {}", error));
        }

        if(workspaceAddr != nullptr) {
            aclrtFree(workspaceAddr);
        }

        return out;
    }


    NPUArray Hypot(const NPUArray& a, const NPUArray& b) {
        auto broadcast = GetBroadcastShape(a, b);

        // 步骤1: 计算a的平方 (a²)
        NPUArray a_squared(a.shape, a.dtype());
        uint64_t a_sq_workspace_size = 0;
        aclOpExecutor* a_sq_executor = nullptr;
        auto error = aclnnMulGetWorkspaceSize(
            a.tensorPtr, a.tensorPtr,
            a_squared.tensorPtr,
            &a_sq_workspace_size,
            &a_sq_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* a_sq_workspace = nullptr;
        if (a_sq_workspace_size != 0ULL) {
            error = aclrtMalloc(&a_sq_workspace, a_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnMul(a_sq_workspace, a_sq_workspace_size, a_sq_executor, nullptr);
        CheckAclnnStatus(error, "aclnnMul error");

        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        // 步骤2: 计算b的平方 (b²)
        NPUArray b_squared(b.shape, b.dtype());
        uint64_t b_sq_workspace_size = 0;
        aclOpExecutor* b_sq_executor = nullptr;
        error = aclnnMulGetWorkspaceSize(
            b.tensorPtr, b.tensorPtr,
            b_squared.tensorPtr,
            &b_sq_workspace_size,
            &b_sq_executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        void* b_sq_workspace = nullptr;
        if (b_sq_workspace_size != 0ULL) {
            error = aclrtMalloc(&b_sq_workspace, b_sq_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnMul(b_sq_workspace, b_sq_workspace_size, b_sq_executor, nullptr);
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

        void* add_workspace = nullptr;
        if (add_workspace_size != 0ULL) {
            error = aclrtMalloc(&add_workspace, add_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnAdd(add_workspace, add_workspace_size, add_executor, nullptr);
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

        void* sqrt_workspace = nullptr;
        if (sqrt_workspace_size != 0ULL) {
            error = aclrtMalloc(&sqrt_workspace, sqrt_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        error = aclnnSqrt(sqrt_workspace, sqrt_workspace_size, sqrt_executor, nullptr);
        CheckAclnnStatus(error, "aclnnSqrt error");

        // 同步设备并释放资源
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        aclDestroyScalar(alpha_scalar);
        if (a_sq_workspace) {
            aclrtFree(a_sq_workspace);
        }
        if (b_sq_workspace) {
            aclrtFree(b_sq_workspace);
        }
        if (add_workspace) {
            aclrtFree(add_workspace);
        }
        if (sqrt_workspace) {
            aclrtFree(sqrt_workspace);
        }

        return result;
    }


    NPUArray Arctan2(const NPUArray& y, const NPUArray& x) {
        auto broadcast = GetBroadcastShape(y, x);

        // 初始化结果数组
        auto dtype = ACL_FLOAT;
        if (x.aclDtype == ACL_DOUBLE || y.aclDtype == ACL_DOUBLE){
            dtype = ACL_DOUBLE;
        }
        NPUArray result(broadcast, dtype);

        // 获取工作空间大小
        uint64_t workspace_size = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnAtan2GetWorkspaceSize(
            y.tensorPtr, x.tensorPtr,
            result.tensorPtr,
            &workspace_size, &executor
        );
        CheckGetWorkspaceSizeAclnnStatus(error);

        // 分配工作空间
        void* workspace = nullptr;
        if (workspace_size != 0ULL) {
            error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }

        // 执行计算
        error = aclnnAtan2(workspace, workspace_size, executor, nullptr);
        CheckAclnnStatus(error, "aclnnAtan2 error");

        // 同步设备并释放资源
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        if (workspace) {
            aclrtFree(workspace);
        }

        return result;
    }


    NPUArray Radians(const NPUArray& x) {
        if (x.aclDtype != ACL_FLOAT && x.aclDtype != ACL_DOUBLE && x.aclDtype != ACL_FLOAT16) {
            throw std::runtime_error("Radians: input must be float, double or float16 type");
        }

        NPUArray result(x.shape, x.aclDtype);

        // 用 aclCreateScalar 创建 π/180 标量，避免 ForeachMulScalar 无 error 检查的问题
        const double rad_factor_d = M_PI / 180.0;
        const float  rad_factor_f = static_cast<float>(rad_factor_d);
        aclScalar* scale = (x.aclDtype == ACL_DOUBLE)
            ? aclCreateScalar(const_cast<double*>(&rad_factor_d), ACL_DOUBLE)
            : aclCreateScalar(const_cast<float*>(&rad_factor_f),  ACL_FLOAT);

        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        auto error = aclnnMulsGetWorkspaceSize(x.tensorPtr, scale, result.tensorPtr,
                                               &workspaceSize, &executor);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(scale);
            throw std::runtime_error(fmt::format("Radians: get workspace size failed, error={}", error));
        }

        void* workspace = nullptr;
        if (workspaceSize > 0) {
            error = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (error != ACL_SUCCESS) {
                aclDestroyScalar(scale);
                throw std::runtime_error("Radians: malloc workspace failed");
            }
        }

        error = aclnnMuls(workspace, workspaceSize, executor, nullptr);
        if (workspace) aclrtFree(workspace);
        aclDestroyScalar(scale);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Radians: computation failed, error={}", error));
        }
        aclrtSynchronizeDevice();

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
        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnMul(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnMul error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        if (workspaceAddr) {
            aclrtFree(workspaceAddr);
        }
        return out;
    }
}
