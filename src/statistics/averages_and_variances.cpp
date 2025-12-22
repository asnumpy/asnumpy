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

#include <asnumpy/statistics/averages_and_variances.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_flatten.h>

#include <cstdint>
#include <cstdio>
#include <fmt/base.h>
#include <fmt/format.h>
#include <optional>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace asnumpy {
    namespace {
        int64_t CalculateTotalElements(const std::vector<int64_t>& shape) {
            int64_t total = 1;
            for (size_t i = 0; i < shape.size(); i++) {
                total *= shape[i];
            }
            return total;
        }

        NPUArray FlattenArray(const NPUArray& a) {
            int64_t totalElements = CalculateTotalElements(a.shape);
            std::vector<int64_t> flatShape = {1, totalElements};
            auto temp = NPUArray(flatShape, a.aclDtype);
            
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor;
            auto error = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize, &executor);
            CheckGetWorkspaceSizeAclnnStatus(error);
            
            void* workspaceAddr = nullptr;
            if (workspaceSize != 0ULL) {
                error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
                CheckMallocAclnnStatus(error);
            }
            
            error = aclnnFlatten(workspaceAddr, workspaceSize, executor, nullptr);
            CheckAclnnStatus(error, "aclnnFlatten error");
            error = aclrtSynchronizeDevice();
            CheckSynchronizeDeviceAclnnStatus(error);
            
            return temp;
        }

        double ExtractScalarValue(const NPUArray& result) {
            py::array x = result.ToNumpy();
            py::dtype dt = x.dtype();
            py::buffer_info buf = x.request();
            
            if (dt.is(py::dtype::of<int>())) {
                int* results = static_cast<int*>(buf.ptr);
                return results[0];
            } 
            else if (dt.is(py::dtype::of<double>())) {
                double* results = static_cast<double*>(buf.ptr);
                return results[0];
            }
            else if (dt.is(py::dtype::of<float>())) {
                float* results = static_cast<float*>(buf.ptr);
                return results[0];
            }
            else {
                throw std::runtime_error("Unsupported array data type!");
            }
        }
    }

    NPUArray Mean(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        int64_t ax = axis;
        if (axis < 0) {
            ax = shape.size() + axis;
        }
        if (keepdims) {
            shape[ax] = 1;
        }
        else {
            shape.erase(shape.begin() + ax);
        }
        std::vector<int64_t> tmp{axis};
        aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnMeanGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnMean(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnMean error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    double Mean(const NPUArray& a) {
        auto temp = FlattenArray(a);
        
        std::vector<int64_t> tmp{1};
        aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
        auto result = NPUArray({1}, a.aclDtype);
        
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnMeanGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        
        void* workspaceAddr = nullptr;
        if (workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        
        error = aclnnMean(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnMean error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        
        return ExtractScalarValue(result);
    }
}

