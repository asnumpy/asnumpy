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
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_gcd.h>

#include <fmt/core.h>
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
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace mul_workspace(mul_workspace_size);

    error = aclnnMul(mul_workspace.get(), mul_workspace_size, mul_executor, nullptr);
    CheckExecuteAclnnStatus(error, "Lcm");

    // 步骤2: 计算x1和x2的绝对值乘积 (|a * b|)
    NPUArray abs_product(shape, out_dtype);
    uint64_t abs_workspace_size = 0;
    aclOpExecutor* abs_executor = nullptr;
    error = aclnnAbsGetWorkspaceSize(
        product.tensorPtr, abs_product.tensorPtr,
        &abs_workspace_size, &abs_executor
    );
    CheckGetWorkspaceSizeAclnnStatus(error);

    AclWorkspace abs_workspace(abs_workspace_size);

    error = aclnnAbs(abs_workspace.get(), abs_workspace_size, abs_executor, nullptr);
    CheckExecuteAclnnStatus(error, "Lcm");

    // 步骤3: 计算x1和x2的最大公约数 (GCD(a, b))
    NPUArray gcd_result = Gcd(x1, x2);  // 复用已实现的Gcd函数

    // 步骤4: 计算LCM = |a*b| / GCD(a,b)
    return ExecuteBinaryOp(
        abs_product,
        gcd_result,                                           
        out_dtype,                                     
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnDivGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnDiv(workspace, workspaceSize, executor, nullptr);
        },
        "Lcm"
    );
}
    

NPUArray Gcd(const NPUArray& x1, const NPUArray& x2, std::optional<py::dtype> dtype) {
    // 初始化结果数组（广播输出形状）
    auto out_dtype = x1.dtype;
    auto shape = GetBroadcastShape(x1, x2);
    auto out = NPUArray(shape, out_dtype);
    if (dtype != std::nullopt) {
        out_dtype = *dtype;
    }
    return ExecuteBinaryOp(
        x1,
        x2,                                           
        out_dtype,                                     
        [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnGcdGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnGcd(workspace, workspaceSize, executor, nullptr);
        },
        "Gcd"
    );
}

}