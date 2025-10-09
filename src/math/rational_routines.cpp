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


#include <asnumpy/math/rational_routines.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_gcd.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Lcm(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 初始化中间结果和最终结果数组
    auto out_dtype = x1.dtype;
    auto acl_dtype = x1.aclDtype;
    auto shape = GetBroadcastShape(x1, x2);
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    auto out = NPUArray(shape, out_dtype);

    // 步骤1: 计算x1和x2的乘积 (a * b)
    NPUArray product(shape, out_dtype);
    uint64_t mul_workspace_size = 0;
    aclOpExecutor* mul_executor = nullptr;
    auto error = aclnnMulGetWorkspaceSize(
        x1.tensorPtr, x2.tensorPtr,
        product.tensorPtr,
        &mul_workspace_size, &mul_executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Lcm: product workspace size failed, error={}", error));
    }

    void* mul_workspace = nullptr;
    if (mul_workspace_size > 0) {
        error = aclrtMalloc(&mul_workspace, mul_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Lcm: product workspace malloc failed, error={}", error));
        }
    }

    error = aclnnMul(mul_workspace, mul_workspace_size, mul_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        throw std::runtime_error(fmt::format("Lcm: product computation failed, error={}", error));
    }

    // 步骤2: 计算x1和x2的绝对值乘积 (|a * b|)
    NPUArray abs_product(shape, out_dtype);
    uint64_t abs_workspace_size = 0;
    aclOpExecutor* abs_executor = nullptr;
    error = aclnnAbsGetWorkspaceSize(
        product.tensorPtr, abs_product.tensorPtr,
        &abs_workspace_size, &abs_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        throw std::runtime_error(fmt::format("Lcm: abs workspace size failed, error={}", error));
    }

    void* abs_workspace = nullptr;
    if (abs_workspace_size > 0) {
        error = aclrtMalloc(&abs_workspace, abs_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(mul_workspace);
            throw std::runtime_error(fmt::format("Lcm: abs workspace malloc failed, error={}", error));
        }
    }

    error = aclnnAbs(abs_workspace, abs_workspace_size, abs_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        throw std::runtime_error(fmt::format("Lcm: abs computation failed, error={}", error));
    }

    // 步骤3: 计算x1和x2的最大公约数 (GCD(a, b))
    NPUArray gcd_result = Gcd(x1, x2);  // 复用已实现的Gcd函数

    // 步骤4: 计算LCM = |a*b| / GCD(a,b)
    NPUArray result(shape, out_dtype);
    uint64_t div_workspace_size = 0;
    aclOpExecutor* div_executor = nullptr;
    error = aclnnDivGetWorkspaceSize(
        abs_product.tensorPtr, gcd_result.tensorPtr,
        result.tensorPtr,
        &div_workspace_size, &div_executor
    );
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        throw std::runtime_error(fmt::format("Lcm: division workspace size failed, error={}", error));
    }

    void* div_workspace = nullptr;
    if (div_workspace_size > 0) {
        error = aclrtMalloc(&div_workspace, div_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclrtFree(mul_workspace);
            aclrtFree(abs_workspace);
            throw std::runtime_error(fmt::format("Lcm: division workspace malloc failed, error={}", error));
        }
    }

    error = aclnnDiv(div_workspace, div_workspace_size, div_executor, nullptr);
    if (error != ACL_SUCCESS) {
        aclrtFree(mul_workspace);
        aclrtFree(abs_workspace);
        aclrtFree(div_workspace);
        throw std::runtime_error(fmt::format("Lcm: division computation failed, error={}", error));
    }

    // 同步设备并释放所有资源
    aclrtSynchronizeDevice();
    aclrtFree(mul_workspace);
    aclrtFree(abs_workspace);
    aclrtFree(div_workspace);

    return result;
}
    

NPUArray Gcd(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 初始化结果数组（广播输出形状）
    auto out_dtype = x1.dtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(shape, out_dtype);
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    NPUArray result(shape, out_dtype);

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnGcdGetWorkspaceSize(
        x1.tensorPtr,
        x2.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Gcd: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Gcd: malloc workspace failed, error={}", error));
        }
    }

    // 执行最大公约数计算
    error = aclnnGcd(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Gcd: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}

}