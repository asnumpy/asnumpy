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


#include <asnumpy/math/sums_products_differences.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_prod.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_flatten.h>
#include <aclnnop/aclnn_reduce_nansum.h>
#include <aclnnop/aclnn_cumsum.h>
#include <aclnnop/aclnn_cumprod.h>
#include <aclnnop/aclnn_nan_to_num.h>
#include <aclnnop/aclnn_linalg_cross.h>

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
    NPUArray Prod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
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
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnProdDimGetWorkspaceSize(a.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnProdDim(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnProdDim error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    double Prod(const NPUArray& a) {
        std::vector<int64_t> shape = {1};
        auto result = NPUArray(shape, a.aclDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnProdGetWorkspaceSize(a.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnProd(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnProd error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);

        py::array temp = result.ToNumpy();
        py::dtype dt = temp.dtype();
        py::buffer_info buf = temp.request();
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
        return 0;
    }

    NPUArray Sum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
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
        auto error = aclnnReduceSumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnReduceSum(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnReduceSum error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    double Sum(const NPUArray& a) {
        std::vector<int64_t> shape = a.shape;
        int64_t pro = 1;
        for (int i=0; i<shape.size(); i++){
            pro = pro * shape[i];
        }
        shape = {1, pro};
        auto temp = NPUArray(shape, a.aclDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnFlatten(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnFlatten error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);
        
        std::vector<int64_t> tmp{1};
        aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
        auto result = NPUArray({1}, a.aclDtype);
        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnReduceSumGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnReduceSum(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnReduceSum error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);
        
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
        return 0;
    }

    NPUArray Nanprod(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
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
        float scalar = 1.0;
        auto temp = NPUArray(a.shape, a.aclDtype);
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, 
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnNanToNum error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);
        
        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnProdDimGetWorkspaceSize(temp.tensorPtr, axis, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnProdDim(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnProdDim error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);
        return result;
    }

    double Nanprod(const NPUArray& a) {
        std::vector<int64_t> shape = {1};
        float scalar = 1.0;
        auto temp = NPUArray(a.shape, a.aclDtype);
        auto result = NPUArray(shape, a.aclDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, 
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnNanToNum error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);
        
        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnProdGetWorkspaceSize(temp.tensorPtr, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnProd(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnProd error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);

        py::array tmp = result.ToNumpy();
        py::dtype dt = tmp.dtype();
        py::buffer_info buf = tmp.request();
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
        return 0;
    }

    NPUArray Nansum(const NPUArray& a, int64_t axis, bool keepdims, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        float scalar = 0.0;
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
        auto error = aclnnReduceNansumGetWorkspaceSize(a.tensorPtr, axis_array, keepdims, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnReduceNansum(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnReduceNansum error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    double Nansum(const NPUArray& a) {
        auto shape = a.shape;
        int64_t pro = 1;
        for (int i=0; i<shape.size(); i++){
            pro = pro * shape[i];
        }
        shape = {1, pro};
        auto temp = NPUArray(shape, a.aclDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnFlatten(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnFlatten error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);
        
        std::vector<int64_t> tmp{1};
        aclIntArray* axis_array = aclCreateIntArray(tmp.data(), tmp.size());
        auto result = NPUArray({1}, a.aclDtype);
        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnReduceNansumGetWorkspaceSize(temp.tensorPtr, axis_array, false, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnReduceNansum(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnReduceNansum error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);

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
        return 0;
    }

    NPUArray Cumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnCumprodGetWorkspaceSize(a.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnCumprod(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnCumprod error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Cumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnCumsumGetWorkspaceSize(a.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnCumsum(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnCumsum error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }

    NPUArray Nancumprod(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        auto axis_scalar = aclCreateScalar(&axis, ACL_INT64);
        float scalar = 1.0;
        auto temp = NPUArray(shape, a.aclDtype);
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, 
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnNanToNum error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);

        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnCumprodGetWorkspaceSize(temp.tensorPtr, axis_scalar, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnCumprod(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnCumprod error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);
        return result;
    }

    NPUArray Nancumsum(const NPUArray& a, int64_t axis, std::optional<py::dtype> dtype) {
        py::dtype outDtype = dtype.has_value() ? dtype.value() : a.dtype;
        auto shape = a.shape;
        float scalar = 0.0;
        auto temp = NPUArray(shape, a.aclDtype);
        auto result = NPUArray(shape, outDtype);
        uint64_t workspaceSize1 = 0;
        aclOpExecutor* executor1;
        auto error1 = aclnnNanToNumGetWorkspaceSize(a.tensorPtr, scalar, 
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), temp.tensorPtr, &workspaceSize1, &executor1);
        CheckGetWorkspaceSizeAclnnStatus(error1);
        void* workspaceAddr1 = nullptr;
        if(workspaceSize1 != 0ULL) {
            error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error1);
        }
        error1 = aclnnNanToNum(workspaceAddr1, workspaceSize1, executor1, nullptr);
        CheckAclnnStatus(error1, "aclnnNanToNum error");
        error1 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error1);

        uint64_t workspaceSize2 = 0;
        aclOpExecutor* executor2;
        auto error2 = aclnnCumsumGetWorkspaceSize(temp.tensorPtr, axis, result.aclDtype, result.tensorPtr, &workspaceSize2, &executor2);
        CheckGetWorkspaceSizeAclnnStatus(error2);
        void* workspaceAddr2 = nullptr;
        if(workspaceSize2 != 0ULL) {
            error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error2);
        }
        error2 = aclnnCumsum(workspaceAddr2, workspaceSize2, executor2, nullptr);
        CheckAclnnStatus(error2, "aclnnCumsum error");
        error2 = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error2);
        return result;
    }

    NPUArray Cross(const NPUArray& a, const NPUArray& b, int64_t axis) {
        //因AOL接口限制，目前本函数功能并不完整，当前效果和numpy.linalg.cross功能一致
        auto broadcast = GetBroadcastShape(a, b);
        auto result = NPUArray(broadcast, a.aclDtype);
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor;
        auto error = aclnnLinalgCrossGetWorkspaceSize(a.tensorPtr, b.tensorPtr, axis, result.tensorPtr, &workspaceSize, &executor);
        CheckGetWorkspaceSizeAclnnStatus(error);
        void* workspaceAddr = nullptr;
        if(workspaceSize != 0ULL) {
            error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CheckMallocAclnnStatus(error);
        }
        error = aclnnLinalgCross(workspaceAddr, workspaceSize, executor, nullptr);
        CheckAclnnStatus(error, "aclnnLinalgCross error");
        error = aclrtSynchronizeDevice();
        CheckSynchronizeDeviceAclnnStatus(error);
        return result;
    }
}