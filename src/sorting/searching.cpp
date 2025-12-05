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

#include <asnumpy/sorting/searching.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_argmax.h>
#include <aclnnop/aclnn_argmin.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace asnumpy {

namespace {
std::pair<std::vector<int64_t>, int64_t> PrepareReduceShape(
    const std::vector<int64_t>& shape, int64_t axis, bool keepdim) {
    if (shape.empty()) {
        throw std::invalid_argument("ArgMax/ArgMin expect input tensor to have at least one dimension.");
    }

    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t normalizedAxis = axis;
    if (normalizedAxis < 0) {
        normalizedAxis += ndim;
    }
    if (normalizedAxis < 0 || normalizedAxis >= ndim) {
        throw std::out_of_range("axis out of range in ArgMax/ArgMin");
    }

    std::vector<int64_t> outputShape = shape;
    if (keepdim) {
        outputShape[normalizedAxis] = 1;
    } else {
        outputShape.erase(outputShape.begin() + normalizedAxis);
    }

    return {outputShape, normalizedAxis};
}

NPUArray RunArgOp(const NPUArray& a, int64_t axis, bool keepdim,
                  aclnnStatus(*getWorkspace)(const aclTensor*, int64_t, bool, aclTensor*, uint64_t*, aclOpExecutor**),
                  aclnnStatus(*runOp)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    auto [shape, normalizedAxis] = PrepareReduceShape(a.shape, axis, keepdim);
    auto out = NPUArray(shape, ACL_INT64);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto error = getWorkspace(a.tensorPtr, normalizedAxis, keepdim, out.tensorPtr, &workspaceSize, &executor);
    CheckGetWorkspaceSizeAclnnStatus(error);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0ULL) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CheckMallocAclnnStatus(error);
    }

    error = runOp(workspaceAddr, workspaceSize, executor, nullptr);
    CheckAclnnStatus(error, "Arg operation execution failed.");

    error = aclrtSynchronizeDevice();
    CheckSynchronizeDeviceAclnnStatus(error);

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}

}  // namespace

NPUArray ArgMax(const NPUArray& a, int64_t axis, bool keepdim) {
    return RunArgOp(
        a, axis, keepdim,
        aclnnArgMaxGetWorkspaceSize,
        aclnnArgMax);
}

NPUArray ArgMin(const NPUArray& a, int64_t axis, bool keepdim) {
    return RunArgOp(
        a, axis, keepdim,
        aclnnArgMinGetWorkspaceSize,
        aclnnArgMin);
}

}
