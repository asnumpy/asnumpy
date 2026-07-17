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

#include <asnumpy/dtypes/dtype_table.hpp>

#include <array>
#include <complex>
#include <fmt/core.h>
#include <stdexcept>

namespace asnumpy::dtypes {

namespace {

struct Entry {
    aclDataType acl;
    int npy_num; // normalized NumPy type number
    int64_t itemsize;
    const char* name;
};

/**
 * The table. Built on first use rather than at static-init time because py::dtype needs a live
 * interpreter.
 *
 * Keyed on normalized_num() rather than dtype identity: pybind11 normalizes equivalent types that
 * carry different type numbers, so np.int64 and np.longlong (both 8-byte signed on LP64) resolve
 * to the same entry. The previous identity-based chain rejected np.longlong outright.
 */
const std::array<Entry, 14>& Table() {
    static const std::array<Entry, 14> table = {{
        {ACL_BOOL, py::dtype::of<bool>().normalized_num(), 1, "bool"},
        {ACL_INT8, py::dtype::of<int8_t>().normalized_num(), 1, "int8"},
        {ACL_INT16, py::dtype::of<int16_t>().normalized_num(), 2, "int16"},
        {ACL_INT32, py::dtype::of<int32_t>().normalized_num(), 4, "int32"},
        {ACL_INT64, py::dtype::of<int64_t>().normalized_num(), 8, "int64"},
        {ACL_UINT8, py::dtype::of<uint8_t>().normalized_num(), 1, "uint8"},
        {ACL_UINT16, py::dtype::of<uint16_t>().normalized_num(), 2, "uint16"},
        {ACL_UINT32, py::dtype::of<uint32_t>().normalized_num(), 4, "uint32"},
        {ACL_UINT64, py::dtype::of<uint64_t>().normalized_num(), 8, "uint64"},
        {ACL_FLOAT16, py::dtype("float16").normalized_num(), 2, "float16"},
        {ACL_FLOAT, py::dtype::of<float>().normalized_num(), 4, "float32"},
        {ACL_DOUBLE, py::dtype::of<double>().normalized_num(), 8, "float64"},
        {ACL_COMPLEX64, py::dtype::of<std::complex<float>>().normalized_num(), 8, "complex64"},
        {ACL_COMPLEX128, py::dtype::of<std::complex<double>>().normalized_num(), 16, "complex128"},
    }};
    return table;
}

const Entry* FindByAcl(aclDataType acl) {
    for (const auto& e : Table()) {
        if (e.acl == acl)
            return &e;
    }
    return nullptr;
}

/// Byte sizes for ACL types outside the supported set. Kept so diagnostics and byte-size math keep
/// working for types that cannot cross into Python.
int64_t UnsupportedItemSize(aclDataType acl) {
    switch (acl) {
    case ACL_BF16:
        return 2;
    case ACL_COMPLEX32:
        return 4;
    // Sub-byte types are byte-aligned host-side. Note ACL packs them multiple-per-byte on device,
    // so this is the host stride, not the device layout.
    case ACL_INT4:
    case ACL_UINT1:
    case ACL_HIFLOAT8:
    case ACL_FLOAT8_E5M2:
    case ACL_FLOAT8_E4M3FN:
    case ACL_FLOAT8_E8M0:
    case ACL_FLOAT6_E3M2:
    case ACL_FLOAT6_E2M3:
    case ACL_FLOAT4_E2M1:
    case ACL_FLOAT4_E1M2:
        return 1;
    case ACL_STRING:
        return sizeof(char*);
    default:
        return -1;
    }
}

const char* UnsupportedName(aclDataType acl) {
    switch (acl) {
    case ACL_BF16:
        return "bfloat16";
    case ACL_COMPLEX32:
        return "complex32";
    case ACL_INT4:
        return "int4";
    case ACL_UINT1:
        return "uint1";
    case ACL_HIFLOAT8:
        return "hifloat8";
    case ACL_FLOAT8_E5M2:
        return "float8_e5m2";
    case ACL_FLOAT8_E4M3FN:
        return "float8_e4m3fn";
    case ACL_FLOAT8_E8M0:
        return "float8_e8m0";
    case ACL_FLOAT6_E3M2:
        return "float6_e3m2";
    case ACL_FLOAT6_E2M3:
        return "float6_e2m3";
    case ACL_FLOAT4_E2M1:
        return "float4_e2m1";
    case ACL_FLOAT4_E1M2:
        return "float4_e1m2";
    case ACL_STRING:
        return "string";
    case ACL_DT_UNDEFINED:
        return "undefined";
    default:
        return "<unknown>";
    }
}

std::string SupportedNames() {
    std::string out;
    for (const auto& e : Table()) {
        if (!out.empty())
            out += ", ";
        out += e.name;
    }
    return out;
}

} // namespace

bool IsSupported(aclDataType acl) { return FindByAcl(acl) != nullptr; }

bool IsInexact(aclDataType acl) {
    switch (acl) {
    case ACL_FLOAT16:
    case ACL_FLOAT:
    case ACL_DOUBLE:
    case ACL_BF16:
    case ACL_COMPLEX32:
    case ACL_COMPLEX64:
    case ACL_COMPLEX128:
        return true;
    default:
        return false;
    }
}

aclDataType AclFromNumpy(const py::dtype& dtype) {
    // Device memory is little-endian and ACL has no byte-order concept, so a byte-swapped array
    // would be reinterpreted rather than converted. Reject it explicitly instead of letting it
    // fall through to a confusing "unsupported dtype".
    const char order = dtype.byteorder();
    if (order == '>') {
        throw std::invalid_argument(
            fmt::format("[dtype_table.cpp](AclFromNumpy) big-endian dtype '{}' is not supported; "
                        "convert with arr.astype(arr.dtype.newbyteorder('<')) first",
                        py::str(dtype).cast<std::string>()));
    }
    if (dtype.has_fields()) {
        throw std::invalid_argument("[dtype_table.cpp](AclFromNumpy) structured dtypes are not supported");
    }

    const int num = dtype.normalized_num();
    for (const auto& e : Table()) {
        if (e.npy_num == num)
            return e.acl;
    }
    throw std::invalid_argument(fmt::format("[dtype_table.cpp](AclFromNumpy) unsupported dtype '{}'; supported: {}",
                                            py::str(dtype).cast<std::string>(), SupportedNames()));
}

py::dtype NumpyFromAcl(aclDataType acl) {
    if (const Entry* e = FindByAcl(acl))
        return py::dtype(e->name);
    throw std::invalid_argument(
        fmt::format("[dtype_table.cpp](NumpyFromAcl) ACL type '{}' has no NumPy equivalent, so it cannot be "
                    "represented as an asnumpy array; supported: {}",
                    UnsupportedName(acl), SupportedNames()));
}

int64_t ItemSize(aclDataType acl) {
    if (const Entry* e = FindByAcl(acl))
        return e->itemsize;
    const int64_t size = UnsupportedItemSize(acl);
    if (size >= 0)
        return size;
    if (acl == ACL_DT_UNDEFINED)
        return 0;
    throw std::invalid_argument(
        fmt::format("[dtype_table.cpp](ItemSize) unknown aclDataType {}", static_cast<int>(acl)));
}

const char* Name(aclDataType acl) {
    if (const Entry* e = FindByAcl(acl))
        return e->name;
    return UnsupportedName(acl);
}

} // namespace asnumpy::dtypes
