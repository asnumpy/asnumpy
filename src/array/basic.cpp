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


#include "asnumpy/array/basic.hpp"
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_ones.h>
#include <aclnnop/aclnn_zero.h>
#include <aclnnop/aclnn_eye.h>
#include <aclnnop/aclnn_arange.h>
#include <fmt/core.h>
#include <fmt/format.h>


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
