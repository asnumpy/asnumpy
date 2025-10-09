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


#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <asnumpy/linalg/product.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>
#include <fmt/base.h>
#include <fmt/format.h>
#include <stdexcept>
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_flatten.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_mul.h"

NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {
	// 考虑到[1x2][1x2]点积形式不能直接用广播 赫赫
	auto broadcast = GetBroadcastShape(x1, x2);
	// throw std::runtime_error(fmt::format("{}",broadcast));
	auto result = NPUArray(broadcast, ACL_FLOAT);
	int8_t use_fp16 = 2;
	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;
	auto error =
		aclnnMatmulGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, use_fp16, &workspaceSize, &executor);
	CheckGetWorkspaceSizeAclnnStatus(error);
	void* workspaceAddr = nullptr;
	if (workspaceSize > 0) {
		error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
		CheckMallocAclnnStatus(error);
	}
	error = aclnnMatmul(workspaceAddr, workspaceSize, executor, nullptr);
	CheckAclnnStatus(error, "aclnnMatmul error");
	error = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(error);
	return result;
}

NPUArray Einsum(const char* subscripts, const std::vector<NPUArray>& operands) {
	// aclnnEinsum目前只支持'abcd,abced->abce'，'a,b->ab'(outer)操作，所以就当一共两个操作数来实现:)
	std::vector<aclTensor*> tmp{operands[0].tensorPtr, operands[1].tensorPtr};
	auto input = aclCreateTensorList(tmp.data(), tmp.size());
	// TODO: reshape广播
	auto broadcast = GetBroadcastShape(operands[0], operands[1]);
	auto result = NPUArray(broadcast, operands[0].dtype);
	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;
	auto error = aclnnEinsumGetWorkspaceSize(input, subscripts, result.tensorPtr, &workspaceSize, &executor);
	CheckGetWorkspaceSizeAclnnStatus(error);
	void* workspaceAddr = nullptr;
	if (workspaceSize > 0) {
		error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
		CheckMallocAclnnStatus(error);
	}
	error = aclnnEinsum(workspaceAddr, workspaceSize, executor, nullptr);
	CheckAclnnStatus(error, "aclnnEinsum error");
	error = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(error);
	return result;
}

NPUArray Matrix_power(const NPUArray& a, int64_t n) {
	auto shape = a.shape;
	auto temp = NPUArray(shape, a.aclDtype);
	auto result = NPUArray(shape, a.aclDtype);
	int absn = 0;
	if (n == 0) {
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor;
		auto error = aclnnEyeGetWorkspaceSize(shape[0], shape[1], result.tensorPtr, &workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(error);
		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			CheckMallocAclnnStatus(error);
		}
		error = aclnnEye(workspaceAddr, workspaceSize, executor, nullptr);
		CheckAclnnStatus(error, "aclnnEye error");
		error = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error);
		return result;
	}

	else if (n < 0) {
		absn = -n;
		uint64_t workspaceSize1 = 0;
		aclOpExecutor* executor1;
		auto error1 = aclnnInverseGetWorkspaceSize(a.tensorPtr, temp.tensorPtr, &workspaceSize1, &executor1);
		CheckGetWorkspaceSizeAclnnStatus(error1);
		void* workspaceAddr1 = nullptr;
		if (workspaceSize1 > 0) {
			error1 = aclrtMalloc(&workspaceAddr1, workspaceSize1, ACL_MEM_MALLOC_HUGE_FIRST);
			CheckMallocAclnnStatus(error1);
		}
		error1 = aclnnInverse(workspaceAddr1, workspaceSize1, executor1, nullptr);
		CheckAclnnStatus(error1, "aclnnInverse error");
		error1 = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error1);
	}
	else {
		absn = n;
		temp = NPUArray(a);
	}

	for (int i = 0; i < absn - 1; i++) {
		auto x = NPUArray(shape, a.aclDtype);
		int8_t use_fp16 = 2;
		uint64_t workspaceSize2 = 0;
		aclOpExecutor* executor2;
		auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, a.tensorPtr, x.tensorPtr, use_fp16, &workspaceSize2,
												  &executor2);
		CheckGetWorkspaceSizeAclnnStatus(error2);
		void* workspaceAddr2 = nullptr;
		if (workspaceSize2 > 0) {
			error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
			CheckMallocAclnnStatus(error2);
		}
		error2 = aclnnMatmul(workspaceAddr2, workspaceSize2, executor2, nullptr);
		CheckAclnnStatus(error2, "aclnnMatmul error");
		error2 = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error2);
		aclTensor* swap = temp.tensorPtr;
		temp.tensorPtr = x.tensorPtr;
		x.tensorPtr = swap;
	}

	int8_t use_fp16 = 2;
	uint64_t workspaceSize2 = 0;
	aclOpExecutor* executor2;
	auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, a.tensorPtr, result.tensorPtr, use_fp16, &workspaceSize2,
											  &executor2);
	CheckGetWorkspaceSizeAclnnStatus(error2);
	void* workspaceAddr2 = nullptr;
	if (workspaceSize2 > 0) {
		error2 = aclrtMalloc(&workspaceAddr2, workspaceSize2, ACL_MEM_MALLOC_HUGE_FIRST);
		CheckMallocAclnnStatus(error2);
	}
	error2 = aclnnMatmul(workspaceAddr2, workspaceSize2, executor2, nullptr);
	CheckAclnnStatus(error2, "aclnnMatmul error");
	error2 = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(error2);
	return result;
}

/**
 * @brief Compute the dot product of two arrays.
 */
NPUArray dot(const NPUArray& a, const NPUArray& b) {
	// case 1: both are scalars
	if (a.shape.size() == 0 && b.shape.size() == 0) {
		auto out = NPUArray({}, a.dtype);
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnDotGetWorkspaceSize failed");

		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (error != ACL_SUCCESS)
				throw std::runtime_error("[product.cpp](dot) aclrtMalloc failed");
		}

		error = aclnnDot(workspaceAddr, workspaceSize, executor, nullptr);
		if (workspaceAddr)
			aclrtFree(workspaceAddr);
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnDot failed");

		error = aclrtSynchronizeDevice();
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclrtSynchronizeDevice failed");

		return out;
	}

	// case 2: 1D · 1D → scalar
	if (a.shape.size() == 1 && b.shape.size() == 1) {
		auto out = NPUArray({}, a.dtype);
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnDotGetWorkspaceSize failed");

		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (error != ACL_SUCCESS)
				throw std::runtime_error("[product.cpp](dot) aclrtMalloc failed");
		}

		error = aclnnDot(workspaceAddr, workspaceSize, executor, nullptr);
		if (workspaceAddr)
			aclrtFree(workspaceAddr);
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnDot failed");

		error = aclrtSynchronizeDevice();
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclrtSynchronizeDevice failed");

		return out;
	}

	// case 3: 2D × 2D → matrix multiply
	if (a.shape.size() == 2 && b.shape.size() == 2) {
		std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
		auto out = NPUArray(out_shape, a.dtype);

		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		int8_t cubeMathType = 0; // KEEP_DTYPE
		auto error =
			aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, &workspaceSize, &executor);

		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnMmGetWorkspaceSize failed");

		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (error != ACL_SUCCESS)
				throw std::runtime_error("[product.cpp](dot) aclrtMalloc failed");
		}

		error = aclnnMm(workspaceAddr, workspaceSize, executor, nullptr);
		if (workspaceAddr)
			aclrtFree(workspaceAddr);
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclnnMm failed");

		error = aclrtSynchronizeDevice();
		if (error != ACL_SUCCESS)
			throw std::runtime_error("[product.cpp](dot) aclrtSynchronizeDevice failed");

		return out;
	}

	// case 4: higher-dim fallback (这里先抛出异常，具体实现可以后续扩展)
	throw std::runtime_error("[product.cpp](dot) Higher-dimensional dot is not yet implemented");
}

/**
 * @brief Compute the dot product of two arrays after flattening.
 *
 * TODO: Currently requires reshape support to convert (1, N) -> (N,).
 *       This project has not implemented reshape yet.
//  */
NPUArray vdot(const NPUArray& a, const NPUArray& b) {
	aclnnStatus ret; // For receiving API return status
	const int64_t num_elements = a.tensorSize;
	// ===================================================================================
	// Step 1: Flatten
	// ===================================================================================
	std::vector<int64_t> flat_shape = {1, num_elements};
	NPUArray a_flat(flat_shape, a.dtype);
	NPUArray b_flat(flat_shape, b.dtype);

	// Flatten 'a'
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		ret = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 0, a_flat.tensorPtr, &workspaceSize, &executor);
		if (ret != ACL_SUCCESS) {
			throw std::runtime_error("[vdot] aclnnFlattenGetWorkspaceSize for 'a' failed.");
		}

		void* workspace_addr = nullptr;
		if (workspaceSize > 0) {
			ret = aclrtMalloc(&workspace_addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (ret != ACL_SUCCESS) {
				throw std::runtime_error("[vdot] aclrtMalloc for 'a' flatten workspace failed.");
			}
		}

		// Pass nullptr for the stream, as in the dot function
		ret = aclnnFlatten(workspace_addr, workspaceSize, executor, nullptr);
		if (workspace_addr) {
			aclrtFree(workspace_addr);
		}
		if (ret != ACL_SUCCESS) {
			throw std::runtime_error("[vdot] aclnnFlatten for 'a' failed.");
		}
	}

	// Flatten 'b'
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		ret = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &workspaceSize, &executor);
		if (ret != ACL_SUCCESS) {
			throw std::runtime_error("[vdot] aclnnFlattenGetWorkspaceSize for 'b' failed.");
		}

		void* workspace_addr = nullptr;
		if (workspaceSize > 0) {
			ret = aclrtMalloc(&workspace_addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (ret != ACL_SUCCESS) {
				throw std::runtime_error("[vdot] aclrtMalloc for 'b' flatten workspace failed.");
			}
		}

		// Pass nullptr for the stream
		ret = aclnnFlatten(workspace_addr, workspaceSize, executor, nullptr);
		if (workspace_addr) {
			aclrtFree(workspace_addr);
		}
		if (ret != ACL_SUCCESS) {
			throw std::runtime_error("[vdot] aclnnFlatten for 'b' failed.");
		}
	}

	// ===================================================================================
	// Step 2: Metadata Reshape
	// ===================================================================================
	std::vector<int64_t> vector_shape = {num_elements};
	std::vector<int64_t> vector_stride = {1};

	aclTensor* a_1d_view =
		aclCreateTensor(vector_shape.data(), vector_shape.size(), a.aclDtype, vector_stride.data(), 0, ACL_FORMAT_ND,
						flat_shape.data(), flat_shape.size(), a_flat.device_address());
	if (!a_1d_view) {
		throw std::runtime_error("[vdot] aclCreateTensor for a_1d_view failed.");
	}

	aclTensor* b_1d_view =
		aclCreateTensor(vector_shape.data(), vector_shape.size(), b.aclDtype, vector_stride.data(), 0, ACL_FORMAT_ND,
						flat_shape.data(), flat_shape.size(), b_flat.device_address());
	if (!b_1d_view) {
		aclDestroyTensor(a_1d_view);
		throw std::runtime_error("[vdot] aclCreateTensor for b_1d_view failed.");
	}

	// ===================================================================================
	// Step 3: Dot Product
	// ===================================================================================
	NPUArray out({}, a.dtype);

	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		ret = aclnnDotGetWorkspaceSize(a_1d_view, b_1d_view, out.tensorPtr, &workspaceSize, &executor);
		if (ret != ACL_SUCCESS) {
			aclDestroyTensor(a_1d_view);
			aclDestroyTensor(b_1d_view);
			throw std::runtime_error("[vdot] aclnnDotGetWorkspaceSize failed.");
		}

		void* workspace_addr = nullptr;
		if (workspaceSize > 0) {
			ret = aclrtMalloc(&workspace_addr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (ret != ACL_SUCCESS) {
				aclDestroyTensor(a_1d_view);
				aclDestroyTensor(b_1d_view);
				throw std::runtime_error("[vdot] aclrtMalloc for dot workspace failed.");
			}
		}

		// Pass nullptr for the stream
		ret = aclnnDot(workspace_addr, workspaceSize, executor, nullptr);
		if (workspace_addr) {
			aclrtFree(workspace_addr);
		}
		if (ret != ACL_SUCCESS) {
			aclDestroyTensor(a_1d_view);
			aclDestroyTensor(b_1d_view);
			throw std::runtime_error("[vdot] aclnnDot failed.");
		}
	}

	// Manually destroy the created tensor views before returning
	aclDestroyTensor(a_1d_view);
	aclDestroyTensor(b_1d_view);

	// Synchronize the device to wait for computation to complete, as in the dot function
	ret = aclrtSynchronizeDevice();
	const char* error_msg = aclGetRecentErrMsg();
	if (ret != ACL_SUCCESS) {
		throw std::runtime_error("[vdot] aclrtSynchronizeDevice failed");
	}

	return out;
}

/**
 * @brief Compute the inner product of two arrays.
 */
NPUArray inner(const NPUArray& a, const NPUArray& b) {
	py::dtype dtype = a.dtype;
	// case 1: 1D × 1D
	if (a.shape.size() == 1 && b.shape.size() == 1) {
		NPUArray out({}, dtype);

		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclnnDotGetWorkspaceSize failed, error = " +
									 std::to_string(error));
		}

		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (error != ACL_SUCCESS) {
				throw std::runtime_error("[product.cpp](inner) aclrtMalloc failed, error = " + std::to_string(error));
			}
		}

		error = aclnnDot(workspaceAddr, workspaceSize, executor, nullptr);
		if (workspaceAddr)
			aclrtFree(workspaceAddr);
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclnnDot failed, error = " + std::to_string(error));
		}

		error = aclrtSynchronizeDevice();
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclrtSynchronizeDevice failed, error = " +
									 std::to_string(error));
		}

		return out;
	}

	// case 2: 2D × 2D
	if (a.shape.size() == 2 && b.shape.size() == 2) {
		std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
		NPUArray out(out_shape, dtype);

		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		int8_t cubeMathType = 0; // KEEP_DTYPE
		auto error =
			aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, &workspaceSize, &executor);
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclnnMmGetWorkspaceSize failed, error = " +
									 std::to_string(error));
		}

		void* workspaceAddr = nullptr;
		if (workspaceSize > 0) {
			error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
			if (error != ACL_SUCCESS) {
				throw std::runtime_error("[product.cpp](inner) aclrtMalloc failed, error = " + std::to_string(error));
			}
		}

		error = aclnnMm(workspaceAddr, workspaceSize, executor, nullptr);
		if (workspaceAddr)
			aclrtFree(workspaceAddr);
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclnnMm failed, error = " + std::to_string(error));
		}

		error = aclrtSynchronizeDevice();
		if (error != ACL_SUCCESS) {
			throw std::runtime_error("[product.cpp](inner) aclrtSynchronizeDevice failed, error = " +
									 std::to_string(error));
		}

		return out;
	}

	// case 3: higher dimensions not supported yet
	throw std::runtime_error("[product.cpp](inner) Only 1D and 2D inputs are supported for now");
}

/**
 * @brief Compute the outer product of two arrays.
 */
NPUArray outer(const NPUArray& a, const NPUArray& b) {
	py::dtype dtype = a.dtype;
	// Step 1: flatten a -> (m, 1)
	auto a_flat = NPUArray({static_cast<int64_t>(a.tensorSize), 1}, a.aclDtype);
	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 1, a_flat.tensorPtr, &ws, &exec);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] FlattenGetWorkspaceSize(a) failed");

		void* wsAddr = nullptr;
		if (ws > 0) {
			err = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
			if (err != ACL_SUCCESS)
				throw std::runtime_error("[outer] aclrtMalloc (a) failed");
		}

		err = aclnnFlatten(wsAddr, ws, exec, nullptr);
		if (wsAddr)
			aclrtFree(wsAddr);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] Flatten(a) exec failed");
	}

	// Step 2: flatten b -> (1, n)
	auto b_flat = NPUArray({1, static_cast<int64_t>(b.tensorSize)}, b.aclDtype);
	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &ws, &exec);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] FlattenGetWorkspaceSize(b) failed");

		void* wsAddr = nullptr;
		if (ws > 0) {
			err = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
			if (err != ACL_SUCCESS)
				throw std::runtime_error("[outer] aclrtMalloc (b) failed");
		}

		err = aclnnFlatten(wsAddr, ws, exec, nullptr);
		if (wsAddr)
			aclrtFree(wsAddr);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] Flatten(b) exec failed");
	}

	// Step 3: elementwise multiply (m,1) * (1,n) -> (m,n)
	std::vector<int64_t> out_shape = {static_cast<int64_t>(a.tensorSize), static_cast<int64_t>(b.tensorSize)};

	NPUArray out(out_shape, dtype);

	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnMulGetWorkspaceSize(a_flat.tensorPtr, b_flat.tensorPtr, out.tensorPtr, &ws, &exec);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] MulGetWorkspaceSize failed");

		void* wsAddr = nullptr;
		if (ws > 0) {
			err = aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST);
			if (err != ACL_SUCCESS)
				throw std::runtime_error("[outer] aclrtMalloc (mul) failed");
		}

		err = aclnnMul(wsAddr, ws, exec, nullptr);
		if (wsAddr)
			aclrtFree(wsAddr);
		if (err != ACL_SUCCESS)
			throw std::runtime_error("[outer] Mul exec failed");
	}

	auto err = aclrtSynchronizeDevice();
	if (err != ACL_SUCCESS)
		throw std::runtime_error("[outer] aclrtSynchronizeDevice failed");

	return out;
}
