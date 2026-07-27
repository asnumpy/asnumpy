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
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_signbit.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <pybind11/pytypes.h>
#include <stdexcept>

namespace asnumpy {

NPUArray Signbit(const NPUArray& x) {
    py::dtype dtype = NPUArray::GetPyDtype(ACL_BOOL);
    return EXECUTE_UNARY_OP(
        x, dtype,
        [](aclTensor* in, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor) {
            return aclnnSignbitGetWorkspaceSize(in, out, workspaceSize, executor);
        },
        [](void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, void* stream) {
            return aclnnSignbit(workspace, workspaceSize, executor, nullptr);
        },
        "Signbit", "aclnnSignbit");
}

NPUArray Ldexp(const NPUArray& x1, const NPUArray& x2) {
    py::object base_scalar = py::float_(2.0);
    NPUArray pow2 = Power(base_scalar, x2);

    // NumPy types ldexp as 'ei->e','fi->f','di->d': x2 is an exponent, not a value, so it never
    // participates in promotion -- the output follows x1 alone. But there is no integer loop
    // either, so an integer x1 widens to float64 rather than truncating (ldexp(1, -1) is 0.5, not
    // 0). Pin the output explicitly; otherwise the internal Multiply promotes against pow2, which
    // Power always produces as float64.
    aclDataType out = dtypes::IsInexact(x1.aclDtype) ? x1.aclDtype : ACL_DOUBLE;
    return Multiply(x1, pow2, dtypes::NumpyFromAcl(out));
}

NPUArray Copysign(const NPUArray& x1, const NPUArray& x2) {
    NPUArray temp1 = Absolute(x1);
    NPUArray temp2 = Sign(x2);
    NPUArray result = Multiply(temp1, temp2);
    return result;
}

} // namespace asnumpy
