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

#pragma once

#include <asnumpy/utils/npu_array.hpp>

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>

#include <cstdint>
#include <vector>

namespace asnumpy {

/**
 * Compute the arithmetic mean over one or more already-normalized axes.
 *
 * `axes` must contain unique, non-negative axes.  An empty vector means a
 * no-op reduction, which is distinct from reducing all dimensions.  The
 * Python layer expands ``axis=None`` to every axis before entering the core so
 * this distinction is never ambiguous.
 */
NPUArray Mean(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, py::dtype compute_dtype,
              py::dtype result_dtype);

/**
 * Compute a mean into an existing array and preserve that Python object's
 * identity and storage.  The mean is accumulated using `compute_dtype`, then
 * written into `out` with a device copy or cast as required.
 */
void MeanOut(const NPUArray& a, const std::vector<int64_t>& axes, bool keepdims, py::dtype compute_dtype,
             NPUArray& out);
} // namespace asnumpy
