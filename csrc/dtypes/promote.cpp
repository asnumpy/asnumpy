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
 ******************************************************************************/

#include <asnumpy/dtypes/promote.hpp>

#include <asnumpy/dtypes/dtype_table.hpp>
#include <asnumpy/utils/cast.hpp>

#include <cstdint>
#include <pybind11/gil_safe_call_once.h>
#include <unordered_map>

namespace asnumpy {

namespace {

/**
 * numpy.result_type, imported once.
 *
 * gil_safe_call_once_and_store rather than a plain function-local static, for two reasons. A bare
 * `static py::object` is destroyed during static destruction, which runs *after* the interpreter
 * finalizes, so its decref touches freed state and segfaults on exit; this utility never destroys
 * the stored object. And the magic-static guard of a bare static would be held across an import
 * that can release the GIL, so two threads first reaching this concurrently could deadlock -- one
 * holding the guard while blocked on the GIL, the other holding the GIL while blocked on the
 * guard. This is exactly what the utility exists to prevent.
 */
const py::object& NumpyResultType() {
    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> storage;
    return storage.call_once_and_store_result([]() { return py::module_::import("numpy").attr("result_type"); })
        .get_stored();
}

uint64_t CacheKey(aclDataType a, aclDataType b) { return (static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b); }

} // namespace

aclDataType ResultType(aclDataType a, aclDataType b) {
    if (a == b)
        return a;

    // Memoized on the dtype pair. Guarded by the GIL, which every caller holds: these run inside
    // pybind11-bound operators, and nothing here releases it.
    static std::unordered_map<uint64_t, aclDataType> cache;
    const uint64_t key = CacheKey(a, b);
    if (auto it = cache.find(key); it != cache.end())
        return it->second;

    // dtypes::NumpyFromAcl throws for ACL types with no NumPy equivalent, which is what we want:
    // a type NumPy cannot name is a type NumPy cannot promote.
    py::object promoted = NumpyResultType()(dtypes::NumpyFromAcl(a), dtypes::NumpyFromAcl(b));
    const aclDataType result = dtypes::AclFromNumpy(promoted.cast<py::dtype>());

    cache.emplace(key, result);
    return result;
}

PromotedOperands::PromotedOperands(const NPUArray& x1, const NPUArray& x2)
    : common_(ResultType(x1.aclDtype, x2.aclDtype)) {
    Materialize(x1, x2);
}

PromotedOperands::PromotedOperands(const NPUArray& x1, const NPUArray& x2, aclDataType common) : common_(common) {
    Materialize(x1, x2);
}

void PromotedOperands::Materialize(const NPUArray& x1, const NPUArray& x2) {
    // Cast only on mismatch. CastTo would deep-copy on a match, and paying two device copies plus
    // two device-wide syncs on every same-dtype op would dwarf the cost of the op itself.
    if (x1.aclDtype == common_) {
        x1_ = &x1;
    } else {
        x1_storage_ = CastTo(x1, common_);
        x1_ = &*x1_storage_;
    }

    if (x2.aclDtype == common_) {
        x2_ = &x2;
    } else {
        x2_storage_ = CastTo(x2, common_);
        x2_ = &*x2_storage_;
    }
}

} // namespace asnumpy
