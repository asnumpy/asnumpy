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


#include <asnumpy/math/arithmetic_operations.hpp>
#include <asnumpy/math/floating_point_routines.hpp>
#include <asnumpy/math/miscellaneous.hpp>
#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_signbit.h>

#include <fmt/base.h>
#include <fmt/format.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Signbit(const NPUArray& x) {
    // 初始化结果数组（形状与输入一致，数据类型为布尔型）
    auto shape = x.shape;
    NPUArray result(shape, ACL_BOOL);  // 布尔型输出（True表示负数）

    // 获取工作空间大小
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnSignbitGetWorkspaceSize(
        x.tensorPtr,
        result.tensorPtr,
        &workspace_size,
        &executor
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Signbit: get workspace size failed, error={}", error));
    }

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0ULL) {
        error = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            throw std::runtime_error(fmt::format("Signbit: malloc workspace failed, error={}", error));
        }
    }

    // 执行符号位检查（检测是否为负数）
    error = aclnnSignbit(
        workspace,
        workspace_size,
        executor,
        nullptr  // 无需回调
    );
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(fmt::format("Signbit: computation failed, error={}", error));
    }

    // 同步设备并释放资源
    aclrtSynchronizeDevice();
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }

    return result;
}

NPUArray Ldexp(const NPUArray& x1, const NPUArray& x2) {
    py::object base_scalar = py::float_(2.0);
    NPUArray pow2 = Power(base_scalar, x2);
    NPUArray result = Multiply(x1, pow2);

    return result;
}

NPUArray Copysign(const NPUArray& x1, const NPUArray& x2) {
    NPUArray temp1 = Absolute(x1);
    NPUArray temp2 = Sign(x2);
    NPUArray result = Multiply(temp1, temp2);

    return result;
}    

}