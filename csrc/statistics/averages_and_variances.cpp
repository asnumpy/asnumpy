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
#include <asnumpy/utils/acl_executor.hpp>
#include <asnumpy/utils/acl_resource.hpp>
#include <asnumpy/utils/cast.hpp>
#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/status_handler.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_mean.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <fmt/core.h>
#include <fmt/format.h>
#include <limits>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace asnumpy {
namespace {

/** Own an aclIntArray for exactly as long as an operator executor needs it. */
class AclIntArrayGuard {
  public:
    explicit AclIntArrayGuard(const std::vector<int64_t>& values)
        : ptr_(aclCreateIntArray(values.empty() ? nullptr : values.data(), values.size())) {
        if (ptr_ == nullptr) {
            throw std::runtime_error(
                "[averages_and_variances.cpp](mean) aclCreateIntArray error = null result");
        }
    }

    ~AclIntArrayGuard() {
        if (ptr_ != nullptr) {
            aclDestroyIntArray(ptr_);
        }
    }

    AclIntArrayGuard(const AclIntArrayGuard&) = delete;
    AclIntArrayGuard& operator=(const AclIntArrayGuard&) = delete;
    AclIntArrayGuard(AclIntArrayGuard&&) = delete;
    AclIntArrayGuard& operator=(AclIntArrayGuard&&) = delete;

    const aclIntArray* get() const noexcept { return ptr_; }

  private:
    aclIntArray* ptr_;
};

/** Own a double aclScalar used as a weak scalar by elementwise operators. */
class AclScalarGuard {
  public:
    explicit AclScalarGuard(double value) : value_(value), ptr_(aclCreateScalar(&value_, ACL_DOUBLE)) {
        if (ptr_ == nullptr) {
            throw std::runtime_error("[averages_and_variances.cpp](mean) aclCreateScalar error = null result");
        }
    }

    ~AclScalarGuard() {
        if (ptr_ != nullptr) {
            aclDestroyScalar(ptr_);
        }
    }

    AclScalarGuard(const AclScalarGuard&) = delete;
    AclScalarGuard& operator=(const AclScalarGuard&) = delete;

    const aclScalar* get() const noexcept { return ptr_; }

  private:
    double value_;
    aclScalar* ptr_;
};

std::vector<int64_t> NormalizeAxes(const std::vector<int64_t>& axes, size_t ndim) {
    std::vector<int64_t> normalized;
    normalized.reserve(axes.size());

    for (int64_t axis : axes) {
        int64_t value = axis;
        if (value < 0) {
            value += static_cast<int64_t>(ndim);
        }
        if (value < 0 || value >= static_cast<int64_t>(ndim)) {
            throw std::out_of_range(fmt::format("axis {} is out of bounds for array of dimension {}", axis, ndim));
        }
        if (std::find(normalized.begin(), normalized.end(), value) != normalized.end()) {
            throw std::invalid_argument("duplicate value in 'axis'");
        }
        normalized.push_back(value);
    }

    return normalized;
}

std::vector<int64_t> ReductionShape(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& axes,
                                    bool keepdims) {
    if (keepdims) {
        std::vector<int64_t> result = input_shape;
        for (int64_t axis : axes) {
            result[static_cast<size_t>(axis)] = 1;
        }
        return result;
    }

    std::vector<int64_t> result;
    result.reserve(input_shape.size() - axes.size());
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (std::find(axes.begin(), axes.end(), static_cast<int64_t>(i)) == axes.end()) {
            result.push_back(input_shape[i]);
        }
    }
    return result;
}

void ValidateOutShape(const NPUArray& out, const std::vector<int64_t>& expected) {
    if (out.shape != expected) {
        throw std::invalid_argument(fmt::format("output shape {} does not match mean result shape {}",
                                                detail::FormatShape(out.shape), detail::FormatShape(expected)));
    }
}

void RunMeanTensor(const aclTensor* input, const std::vector<int64_t>& axes, bool keepdims,
                   aclDataType compute_dtype, NPUArray& out) {
    AclIntArrayGuard axis_array(axes);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnMeanGetWorkspaceSize(input, axis_array.get(), keepdims, compute_dtype, out.tensorPtr,
                                           &workspace_size, &executor);
    ACLNN_CHECK(error, "aclnnMeanGetWorkspaceSize");

    AclWorkspace workspace(workspace_size);
    error = aclnnMean(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "aclnnMean");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "aclnnMean: aclrtSynchronizeDevice");
    LOG_INFO("aclnnMean completed");
}

void RunMean(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, NPUArray& out) {
    LOG_DEBUG("aclnnMean start: input_shape={}, tensorSize={}, input_dtype={}, axes={}, keepdims={}, output_dtype={}",
              detail::FormatShape(a.shape), a.tensorSize, AclDtypeName(a.aclDtype), detail::FormatShape(axes),
              keepdims, AclDtypeName(out.aclDtype));
    RunMeanTensor(a.tensorPtr, axes, keepdims, out.aclDtype, out);
}

void RunAddZero(const NPUArray& input, NPUArray& out) {
    AclScalarGuard zero(0.0);
    AclScalarGuard one(1.0);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error = aclnnAddsGetWorkspaceSize(input.tensorPtr, zero.get(), one.get(), out.tensorPtr,
                                           &workspace_size, &executor);
    ACLNN_CHECK(error, "mean axis=(): aclnnAddsGetWorkspaceSize");

    AclWorkspace workspace(workspace_size);
    error = aclnnAdds(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "mean axis=(): aclnnAdds");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "mean axis=(): aclrtSynchronizeDevice");
}

void CopyInto(const NPUArray& input, NPUArray& out) {
    if (input.shape != out.shape || input.aclDtype != out.aclDtype) {
        throw std::invalid_argument("device copy requires identical shape and dtype");
    }
    if (input.tensorSize == 0 || input.device_address() == out.device_address()) {
        return;
    }

    const size_t bytes = input.tensorSize * NPUArray::GetDataTypeSize(input.aclDtype);
    auto error = aclrtMemcpy(out.device_address(), bytes, input.device_address(), bytes, ACL_MEMCPY_DEVICE_TO_DEVICE);
    ACL_RT_CHECK(error, "mean out: aclrtMemcpy");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "mean out: aclrtSynchronizeDevice");
}

void CastInto(const NPUArray& input, NPUArray& out) {
    if (input.shape != out.shape) {
        throw std::invalid_argument("device cast requires identical shapes");
    }
    if (input.aclDtype == out.aclDtype) {
        CopyInto(input, out);
        return;
    }

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto error =
        aclnnCastGetWorkspaceSize(input.tensorPtr, out.aclDtype, out.tensorPtr, &workspace_size, &executor);
    ACLNN_CHECK(error, "mean out: aclnnCastGetWorkspaceSize");

    AclWorkspace workspace(workspace_size);
    error = aclnnCast(workspace.get(), workspace.size(), executor, nullptr);
    ACLNN_CHECK(error, "mean out: aclnnCast");
    error = aclrtSynchronizeDevice();
    ACL_RT_CHECK(error, "mean out: aclrtSynchronizeDevice");
}

bool ReducesEmptyExtent(const NPUArray& a, const std::vector<int64_t>& axes) {
    return std::any_of(axes.begin(), axes.end(), [&a](int64_t axis) {
        return a.shape[static_cast<size_t>(axis)] == 0;
    });
}

NPUArray MakeNanArray(const std::vector<int64_t>& shape, py::dtype dtype) {
    py::array host(dtype, shape);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const aclDataType acl_dtype = NPUArray::GetACLDataType(dtype);
    if (acl_dtype == ACL_COMPLEX64 || acl_dtype == ACL_COMPLEX128) {
        host.attr("fill")(py::cast(std::complex<double>(nan, nan)));
    } else {
        host.attr("fill")(py::float_(nan));
    }
    return NPUArray::FromNumpy(std::move(host));
}

void ComputeMeanInto(const NPUArray& input, const std::vector<int64_t>& axes, bool keepdims, NPUArray& out) {
    if (input.aclDtype != out.aclDtype || out.shape != ReductionShape(input.shape, axes, keepdims)) {
        throw std::invalid_argument("mean compute input/output metadata mismatch");
    }
    if (out.tensorSize == 0) {
        return;
    }

    if (axes.empty()) {
        // CANN treats an empty dimension array as "all dimensions".  Adding a
        // positive zero performs the required no-op arithmetic without invoking
        // that special case; unlike a raw copy, it canonicalizes negative zero.
        RunAddZero(input, out);
        return;
    }

    if (ReducesEmptyExtent(input, axes)) {
        auto nan_result = MakeNanArray(out.shape, out.dtype);
        CopyInto(nan_result, out);
        return;
    }

    RunMean(input, axes, keepdims, out);
}

NPUArray ComputeMean(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims,
                     aclDataType compute_dtype) {
    auto result = NPUArray(ReductionShape(a.shape, axes, keepdims), compute_dtype);
    if (a.aclDtype == compute_dtype) {
        ComputeMeanInto(a, axes, keepdims, result);
    } else {
        // Explicitly convert before reducing: CANN's dtype argument describes
        // output storage and does not by itself guarantee accumulator precision.
        auto converted = CastTo(a, compute_dtype);
        ComputeMeanInto(converted, axes, keepdims, result);
    }
    return result;
}

NPUArray MeanNormalized(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims,
                        aclDataType compute_dtype, aclDataType result_dtype) {
    auto computed = ComputeMean(a, axes, keepdims, compute_dtype);
    if (computed.aclDtype == result_dtype) {
        return computed;
    }
    return CastTo(computed, result_dtype);
}

void MeanNormalizedOut(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims,
                       aclDataType compute_dtype, NPUArray& out) {
    const auto expected_shape = ReductionShape(a.shape, axes, keepdims);
    ValidateOutShape(out, expected_shape);

    const bool aliases_input =
        &a == &out || (a.device_address() != nullptr && a.device_address() == out.device_address());

    // Write directly into the caller's buffer whenever its dtype is also the
    // requested accumulator dtype.  Aliasing is handled through a temporary:
    // reduction kernels are not guaranteed to support overlapping input/output.
    if (!aliases_input && out.aclDtype == compute_dtype) {
        if (a.aclDtype == compute_dtype) {
            ComputeMeanInto(a, axes, keepdims, out);
        } else {
            auto converted = CastTo(a, compute_dtype);
            ComputeMeanInto(converted, axes, keepdims, out);
        }
        return;
    }

    auto computed = ComputeMean(a, axes, keepdims, compute_dtype);
    CastInto(computed, out);
}
} // namespace

NPUArray Mean(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, py::dtype compute_dtype,
              py::dtype result_dtype) {
    const auto normalized = NormalizeAxes(axes, a.shape.size());
    return MeanNormalized(a, normalized, keepdims, NPUArray::GetACLDataType(compute_dtype),
                          NPUArray::GetACLDataType(result_dtype));
}

void MeanOut(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, py::dtype compute_dtype,
             NPUArray& out) {
    const auto normalized = NormalizeAxes(axes, a.shape.size());
    MeanNormalizedOut(a, normalized, keepdims, NPUArray::GetACLDataType(compute_dtype), out);
}
} // namespace asnumpy
