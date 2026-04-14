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
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/acl_executor.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <stdexcept>
#include <cstring>
#include "aclnnop/aclnn_dot.h"
#include "aclnnop/aclnn_flatten.h"
#include "aclnnop/aclnn_mm.h"
#include "aclnnop/aclnn_mul.h"

using namespace asnumpy;

NPUArray Matmul(const NPUArray& x1, const NPUArray& x2) {
	// 考虑到[1x2][1x2]点积形式不能直接用广播 赫赫
	auto broadcast = GetBroadcastShape(x1, x2);
	// throw std::runtime_error(fmt::format("{}",broadcast));
	auto result = NPUArray(broadcast, ACL_FLOAT);
	int8_t use_fp16 = 2;
	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;
	auto error = aclnnMatmulGetWorkspaceSize(x1.tensorPtr, x2.tensorPtr, result.tensorPtr, use_fp16, 
		&workspaceSize, &executor);
	CheckGetWorkspaceSizeAclnnStatus(error);
	AclWorkspace workspace(workspaceSize);
	error = aclnnMatmul(workspace.get(), workspaceSize, executor, nullptr);
	CheckAclnnStatus(error, "aclnnMatmul error");
	error = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(error);
	return result;
}

NPUArray Einsum(const char* subscripts, const std::vector<NPUArray>& operands) {
	// aclnnEinsum目前只支持'abcd,abced->abce'，'a,b->ab'(outer)操作，所以就当一共两个操作数来实现:)
	std::vector<aclTensor*> tmp{operands[0].tensorPtr, operands[1].tensorPtr};
	auto input = aclCreateTensorList(tmp.data(), tmp.size());
	// einsum输出shape处理
	std::vector<int64_t> shape;
	if (!std::strcmp(subscripts, "a,b->ab")) {
		shape = {operands[0].shape[0], operands[1].shape[0]};
	}
	else {
		shape = {operands[1].shape[0], operands[1].shape[1], operands[1].shape[2], operands[1].shape[4]};
	}
	auto result = NPUArray(shape, operands[0].dtype);
	uint64_t workspaceSize = 0;
	aclOpExecutor* executor;
	auto error = aclnnEinsumGetWorkspaceSize(input, subscripts, result.tensorPtr, &workspaceSize, &executor);
	auto tensorListGuard = [&input]() { if(input) aclDestroyTensorList(input); };
	CheckGetWorkspaceSizeAclnnStatus(error);
	AclWorkspace workspace(workspaceSize);
	error = aclnnEinsum(workspace.get(), workspaceSize, executor, nullptr);
	CheckAclnnStatus(error, "aclnnEinsum error");
	error = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(error);
	tensorListGuard();
	return result;
}

NPUArray Matrix_power(const NPUArray& a, int64_t n) {
	auto shape = a.shape;
	auto temp = NPUArray(shape, a.aclDtype);
	auto ax = NPUArray(shape, a.aclDtype);
	auto result = NPUArray(shape, a.aclDtype);
	int absn = 0;
	if (n == 0) {
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor;
		auto error = aclnnEyeGetWorkspaceSize(shape[0], shape[1], result.tensorPtr, &workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(error);
		AclWorkspace workspace(workspaceSize);
		error = aclnnEye(workspace.get(), workspaceSize, executor, nullptr);
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
		AclWorkspace workspace1(workspaceSize1);
		error1 = aclnnInverse(workspace1.get(), workspaceSize1, executor1, nullptr);
		CheckAclnnStatus(error1, "aclnnInverse error");
		error1 = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error1);
		ax = NPUArray(temp);
	}
	else {
		absn = n;
		temp = NPUArray(a);
		ax = NPUArray(a);
	}

	if (absn == 1) {
		return temp;
	}

	for (int i = 1; i < absn - 1; i++) {
		auto x = NPUArray(shape, a.aclDtype);
		int8_t use_fp16 = 2;
		uint64_t workspaceSize2 = 0;
		aclOpExecutor* executor2;
		auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, ax.tensorPtr, x.tensorPtr, use_fp16, &workspaceSize2,
												  &executor2);
		CheckGetWorkspaceSizeAclnnStatus(error2);
		AclWorkspace workspace2(workspaceSize2);
		error2 = aclnnMatmul(workspace2.get(), workspaceSize2, executor2, nullptr);
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
	auto error2 = aclnnMatmulGetWorkspaceSize(temp.tensorPtr, ax.tensorPtr, result.tensorPtr, use_fp16, &workspaceSize2,
											  &executor2);
	CheckGetWorkspaceSizeAclnnStatus(error2);
	AclWorkspace workspace2(workspaceSize2);
	error2 = aclnnMatmul(workspace2.get(), workspaceSize2, executor2, nullptr);
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
		return ExecuteBinaryOp(
            a,
			b,
            a.dtype,
            [](aclTensor* in1, aclTensor* in2, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
                return aclnnDotGetWorkspaceSize(in1, in2, out, workspaceSize, executor);
            },
            [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
                return aclnnDot(workspace, workspaceSize, executor, nullptr);
            },
            "dot"
        );
	}

	// case 2: 1D · 1D → scalar
	if (a.shape.size() == 1 && b.shape.size() == 1) {
		auto out = NPUArray({}, a.dtype);
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		auto error = aclnnDotGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, &workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(error);

		AclWorkspace workspace(workspaceSize);

		error = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(error, "dot");

		error = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error);

		return out;
	}

	// case 3: 2D × 2D → matrix multiply
	if (a.shape.size() == 2 && b.shape.size() == 2) {
		std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
		auto out = NPUArray(out_shape, a.dtype);

		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		int8_t cubeMathType = 0; // KEEP_DTYPE
		auto error = aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, 
			&workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(error);

		AclWorkspace workspace(workspaceSize);

		error = aclnnMm(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(error, "dot");

		error = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error);

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
		CheckGetWorkspaceSizeAclnnStatus(ret);

		AclWorkspace workspace(workspaceSize);

		// Pass nullptr for the stream, as in the dot function
		ret = aclnnFlatten(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(ret, "vdot");
	}

	// Flatten 'b'
	{
		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		ret = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(ret);

		AclWorkspace workspace(workspaceSize);

		// Pass nullptr for the stream
		ret = aclnnFlatten(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(ret, "vdot");
	}

	// ===================================================================================
	// Step 2: Metadata Reshape
	// ===================================================================================
	std::vector<int64_t> vector_shape = {num_elements};
	std::vector<int64_t> vector_stride = {1};

	aclTensor* a_1d_view = aclCreateTensor(vector_shape.data(), vector_shape.size(), a.aclDtype, vector_stride.data(), 
						0, ACL_FORMAT_ND, flat_shape.data(), flat_shape.size(), a_flat.device_address());
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
		CheckGetWorkspaceSizeAclnnStatus(ret);

		AclWorkspace workspace(workspaceSize);

		// Pass nullptr for the stream
		ret = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(ret, "vdot");
	}

	// Manually destroy the created tensor views before returning
	aclDestroyTensor(a_1d_view);
	aclDestroyTensor(b_1d_view);

	// Synchronize the device to wait for computation to complete, as in the dot function
	ret = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(ret);
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
		CheckGetWorkspaceSizeAclnnStatus(error);

		AclWorkspace workspace(workspaceSize);

		error = aclnnDot(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(error, "inner");

		error = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error);
		return out;
	}

	// case 2: 2D × 2D
	if (a.shape.size() == 2 && b.shape.size() == 2) {
		std::vector<int64_t> out_shape = {a.shape[0], b.shape[1]};
		NPUArray out(out_shape, dtype);

		uint64_t workspaceSize = 0;
		aclOpExecutor* executor = nullptr;
		int8_t cubeMathType = 0; // KEEP_DTYPE
		auto error = aclnnMmGetWorkspaceSize(a.tensorPtr, b.tensorPtr, out.tensorPtr, cubeMathType, 
			&workspaceSize, &executor);
		CheckGetWorkspaceSizeAclnnStatus(error);

		AclWorkspace workspace(workspaceSize);

		error = aclnnMm(workspace.get(), workspaceSize, executor, nullptr);
		CheckExecuteAclnnStatus(error, "inner");

		error = aclrtSynchronizeDevice();
		CheckSynchronizeDeviceAclnnStatus(error);
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
	// Step 1: flatten a -> (m, 1) 有bug，aclnnFlatten无法展开为(m, 1)格式
	auto a_flat = NPUArray({static_cast<int64_t>(a.tensorSize), 1}, a.aclDtype);
	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnFlattenGetWorkspaceSize(a.tensorPtr, 1, a_flat.tensorPtr, &ws, &exec);
		CheckGetWorkspaceSizeAclnnStatus(err);

		AclWorkspace workspace(ws);

		err = aclnnFlatten(workspace.get(), ws, exec, nullptr);
		CheckExecuteAclnnStatus(err, "outer");
	}

	// Step 2: flatten b -> (1, n)
	auto b_flat = NPUArray({1, static_cast<int64_t>(b.tensorSize)}, b.aclDtype);
	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnFlattenGetWorkspaceSize(b.tensorPtr, 0, b_flat.tensorPtr, &ws, &exec);
		CheckGetWorkspaceSizeAclnnStatus(err);

		AclWorkspace workspace(ws);

		err = aclnnFlatten(workspace.get(), ws, exec, nullptr);
		CheckExecuteAclnnStatus(err, "outer");
	}

	// Step 3: elementwise multiply (m,1) * (1,n) -> (m,n)
	std::vector<int64_t> out_shape = {static_cast<int64_t>(a.tensorSize), static_cast<int64_t>(b.tensorSize)};

	NPUArray out(out_shape, dtype);

	{
		uint64_t ws = 0;
		aclOpExecutor* exec = nullptr;
		auto err = aclnnMulGetWorkspaceSize(a_flat.tensorPtr, b_flat.tensorPtr, out.tensorPtr, &ws, &exec);
		CheckGetWorkspaceSizeAclnnStatus(err);

		AclWorkspace workspace(ws);

		err = aclnnMul(workspace.get(), ws, exec, nullptr);
		CheckExecuteAclnnStatus(err, "outer");
	}

	auto err = aclrtSynchronizeDevice();
	CheckSynchronizeDeviceAclnnStatus(err);

	return out;
}
