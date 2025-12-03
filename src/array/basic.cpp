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


#include <asnumpy/array/basic.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <asnumpy/utils/npu_scalar.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <fmt/core.h>
#include <fmt/format.h>

#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_ones.h>
#include <aclnnop/aclnn_zero.h>
#include <aclnnop/aclnn_eye.h>
#include <aclnnop/aclnn_arange.h>
#include <aclnnop/aclnn_linspace.h>
#include <fmt/core.h>
#include <fmt/format.h>


namespace asnumpy {

NPUArray Empty(const std::vector<int64_t>& shape, py::dtype dtype) {
    try {
        return NPUArray(shape, dtype);
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](empty) NPUArray construction error = {}", e.what()));
    }
}

NPUArray EmptyLike(const NPUArray& prototype, py::dtype dtype) {
    try {
        // 若未指定dtype，使用原型数组的dtype
        py::dtype target_dtype = dtype.is_none() ? prototype.dtype() : dtype;
        // 基于原型的形状和目标dtype创建空数组
        return NPUArray(prototype.shape, target_dtype);
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("[creation.cpp](empty_like) NPUArray construction error = {}", e.what()));
    }
}


NPUArray Zeros(const std::vector<int64_t>& shape, py::dtype dtype) {
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZeroGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](zeros) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceZero(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZero error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}

NPUArray Zeros_like(const NPUArray& other, py::dtype dtype) {
    auto array = NPUArray(other.shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZeroGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](zeros) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceZero(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclnnInplaceZero error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](zeros) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}

NPUArray Full(const std::vector<int64_t>& shape, const py::object& value, py::dtype dtype) {
    auto array = NPUArray(shape, dtype);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[creation.cpp](full) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[creation.cpp](full) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalarGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](full) Invalid workspaceSize: {}", workspaceSize));
    // 3. 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalar error = {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtSynchronizeDevice error = {}",error));
    // 6. 释放
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(scalar);
    return array;
}

NPUArray Full_like(const NPUArray& other, const py::object& value, py::dtype dtype) {
    auto array = NPUArray(other.shape, dtype);
    double valueDouble = 0;
    if (value.is_none()) {
        throw std::runtime_error("[creation.cpp](full) Input is None");
    }
    try {
        valueDouble = py::cast<double>(value);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[creation.cpp](full) Conversion error: " + std::string(e.what()));
    }
    aclScalar* scalar = CreateScalar(valueDouble, array.aclDtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceFillScalarGetWorkspaceSize(array.tensorPtr, scalar, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalarGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](full) Invalid workspaceSize: {}", workspaceSize));
    // 3. 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtMalloc error = {}",error));
    }
    error = aclnnInplaceFillScalar(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclnnInplaceFillScalar error = {}", error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](full) aclrtSynchronizeDevice error = {}",error));
    // 6. 释放
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    aclDestroyScalar(scalar);
    return array;
}

NPUArray Eye(int64_t n, py::dtype dtype) {
    auto array = NPUArray({n, n}, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnEyeGetWorkspaceSize(n, n, array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEyeGetWorkspaceSize error = {}",error));
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[creation.cpp](eye) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtMalloc error = {}",error));
    }
    error = aclnnEye(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclnnEye error = {}",error));
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) throw std::runtime_error(fmt::format("[creation.cpp](eye) aclrtSynchronizeDevice error = {}",error));
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}

NPUArray Ones(const std::vector<int64_t>& shape, py::dtype dtype) {
    auto array = NPUArray(shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones) aclnnInplaceOneGetWorkspaceSize error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 检查workspaceSize是否有效
    if(workspaceSize < 0) throw std::runtime_error(fmt::format("[basic.cpp](ones) Invalid workspaceSize: {}", workspaceSize));
    // 申请工作空间
    void *workspaceAddr = nullptr;
    if(workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if(error != ACL_SUCCESS) {
            std::string error_msg = fmt::format("[basic.cpp](ones) aclrtMalloc error = {}", error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
                error_msg += std::string(" - ") + detailed_msg;
            }
            throw std::runtime_error(error_msg);
        }
    }
    error = aclnnInplaceOne(workspaceAddr, workspaceSize, executor, nullptr);
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones) aclnnInplaceOne error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    error = aclrtSynchronizeDevice();
    if(error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones) aclrtSynchronizeDevice error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    // 执行结束后释放工作空间
    if(workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}


NPUArray Identity(int64_t n, py::dtype dtype) {
    auto array = NPUArray({n, n}, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;

    auto error = aclnnEyeGetWorkspaceSize(n, n,  // 行, 列, 对角线偏移(k=0)
                                          array.tensorPtr,
                                          &workspaceSize,
                                          &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](identity) aclnnEyeGetWorkspaceSize error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }

    if (workspaceSize < 0) {
        throw std::runtime_error(fmt::format("[basic.cpp](identity) Invalid workspaceSize: {}", workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = fmt::format("[basic.cpp](identity) aclrtMalloc error = {}", error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
                error_msg += std::string(" - ") + detailed_msg;
            }
            throw std::runtime_error(error_msg);
        }
    }

    error = aclnnEye(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](identity) aclnnEye error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](identity) aclrtSynchronizeDevice error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg != nullptr && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }

    if (workspaceAddr != nullptr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}

NPUArray ones_like(const NPUArray& other, py::dtype dtype) {
    auto array = NPUArray(other.shape, dtype);
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    auto error = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspaceSize, &executor);
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones_like) aclnnInplaceOneGetWorkspaceSize error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceSize < 0) {
        throw std::runtime_error(fmt::format("[basic.cpp](ones_like) Invalid workspaceSize: {}", workspaceSize));
    }
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string error_msg = fmt::format("[basic.cpp](ones_like) aclrtMalloc error = {}", error);
            const char* detailed_msg = aclGetRecentErrMsg();
            if (detailed_msg && strlen(detailed_msg) > 0) {
                error_msg += std::string(" - ") + detailed_msg;
            }
            throw std::runtime_error(error_msg);
        }
    }
    error = aclnnInplaceOne(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones_like) aclnnInplaceOne error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        std::string error_msg = fmt::format("[basic.cpp](ones_like) aclrtSynchronizeDevice error = {}", error);
        const char* detailed_msg = aclGetRecentErrMsg();
        if (detailed_msg && strlen(detailed_msg) > 0) {
            error_msg += std::string(" - ") + detailed_msg;
        }
        throw std::runtime_error(error_msg);
    }
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }
    return array;
}

NPUArray Linspace(const py::object& start, const py::object& end, const py::object& steps, const py::object& dtype) {
    double start_val = 0.0, end_val = 0.0;
    int64_t steps_val = 0;

    try {
        start_val = py::cast<double>(start);
        end_val   = py::cast<double>(end);
        steps_val = py::cast<int64_t>(steps);
    } catch (const py::cast_error& e) {
        throw std::runtime_error("[creation.cpp](linspace) Invalid start/end/steps type: " +
                                 std::string(e.what()));
    }

    if (steps_val <= 0) {
        throw std::runtime_error("[creation.cpp](linspace) steps must be > 0.");
    }

    py::dtype final_dtype;

    if (!dtype.is_none()) {
        try {
            final_dtype = py::dtype(dtype);
        } catch (...) {
            if (py::isinstance<py::str>(dtype)) {
                final_dtype = py::dtype(py::str(dtype));
            }
            else if (py::hasattr(dtype, "__name__")) {
                try {
                    auto numpy = py::module_::import("numpy");
                    final_dtype = numpy.attr("dtype")(dtype);
                } catch (...) {
                    try {
                        auto numpy = py::module_::import("numpy");
                        final_dtype = numpy.attr("dtype")(dtype);
                    } catch (...) {
                        throw std::runtime_error("[creation.cpp](linspace) Failed to create dtype from numpy type: " +
                                                 std::string(py::str(dtype)));
                    }
                }
            }
            else if (py::hasattr(dtype, "dtype")) {
                final_dtype = dtype.attr("dtype");
            }
            else {
                try {
                    std::string dtype_str = py::cast<std::string>(dtype);
                    final_dtype = py::dtype(dtype_str);
                } catch (...) {
                    throw std::runtime_error("[creation.cpp](linspace) Unsupported dtype parameter type: " +
                                             std::string(py::str(dtype)));
                }
            }
        }
    } else {
        final_dtype = py::dtype::of<double>();
    }

    try {
        if (final_dtype.is(py::dtype::of<int64_t>())) {
            final_dtype = py::dtype::of<int32_t>();
        }
    } catch (...) {
        // ignore
    }

    std::vector<int64_t> out_shape = { steps_val };
    NPUArray out(out_shape, final_dtype);

    if (out.tensorPtr == nullptr) {
        throw std::runtime_error("[creation.cpp](linspace) out.tensorPtr is null, failed to allocate output tensor");
    }

    aclScalar* acl_start = nullptr;
    aclScalar* acl_end   = nullptr;

    try {
        acl_start = aclCreateScalar(&start_val, ACL_DOUBLE);
        acl_end   = aclCreateScalar(&end_val,   ACL_DOUBLE);
    } catch (...) {
        if (acl_start) aclDestroyScalar(acl_start);
        if (acl_end)   aclDestroyScalar(acl_end);
        throw std::runtime_error("[creation.cpp](linspace) Failed to create ACL scalars.");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    auto error = aclnnLinspaceGetWorkspaceSize(
        acl_start,
        acl_end,
        steps_val,
        out.tensorPtr,
        &workspaceSize,
        &executor);

    if (error != ACL_SUCCESS) {
        aclDestroyScalar(acl_start);
        aclDestroyScalar(acl_end);
        std::string msg = "[creation.cpp](linspace) aclnnLinspaceGetWorkspaceSize error = " +
                          std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    if (workspaceSize < 0) {
        aclDestroyScalar(acl_start);
        aclDestroyScalar(acl_end);
        throw std::runtime_error("[creation.cpp](linspace) Invalid workspaceSize: " +
                                 std::to_string(workspaceSize));
    }

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            aclDestroyScalar(acl_start);
            aclDestroyScalar(acl_end);
            std::string msg = "[creation.cpp](linspace) aclrtMalloc error = " + std::to_string(error);
            const char* detail = aclGetRecentErrMsg();
            if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
            throw std::runtime_error(msg);
        }
    }

    error = aclnnLinspace(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_start);
        aclDestroyScalar(acl_end);
        std::string msg = "[creation.cpp](linspace) aclnnLinspace error = " + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        aclDestroyScalar(acl_start);
        aclDestroyScalar(acl_end);
        std::string msg = "[creation.cpp](linspace) aclrtSynchronizeDevice error = " + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    if (workspaceAddr) aclrtFree(workspaceAddr);
    aclDestroyScalar(acl_start);
    aclDestroyScalar(acl_end);

    return out;
}

}
