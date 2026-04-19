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

#include <aclnn/acl_meta.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <asnumpy/utils/npu_array.hpp>
#include <asnumpy/utils/tensor_descriptor_cache.hpp>

namespace asnumpy::utils {

struct ExecutorCacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t builds = 0;
    size_t repeatable_failures = 0;
    size_t rebind_failures = 0;
    size_t cached_count = 0;
};

class ExecutorCache {
public:
    struct Handle {
        aclOpExecutor* executor = nullptr;
        uint64_t workspace_size = 0;
        std::vector<std::shared_ptr<void>> retained_args;
    };

    static ExecutorCache& instance();

    Handle prepare_inplace_zero(const NPUArray& array);
    Handle prepare_inplace_one(const NPUArray& array);
    Handle prepare_binary_add(
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output);
    Handle prepare_binary_mul(
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output);
    Handle prepare_reduce_sum(
        const NPUArray& input,
        const NPUArray& output,
        const std::vector<int64_t>& axes,
        bool keepdims);
    void clear() noexcept;
    void reset_stats() noexcept;
    ExecutorCacheStats stats() const;

private:
    ExecutorCache() = default;

    enum class OpKind : uint8_t {
        InplaceZero = 0,
        InplaceOne = 1,
        ReduceSum = 2,
        BinaryAdd = 3,
        BinaryMul = 4,
    };

    struct Key {
        OpKind op_kind = OpKind::InplaceZero;
        TensorDescriptorMeta primary_meta;
        TensorDescriptorMeta secondary_meta;
        TensorDescriptorMeta tertiary_meta;
        std::vector<int64_t> int_array_arg;
        bool bool_arg = false;

        bool operator==(const Key& other) const noexcept;
    };

    struct KeyHash {
        size_t operator()(const Key& key) const noexcept;
    };

    struct Entry {
        aclOpExecutor* executor = nullptr;
        uint64_t workspace_size = 0;
        std::vector<std::shared_ptr<void>> retained_args;
    };

    Handle build_inplace_zero(const NPUArray& array, const Key& key);
    Handle build_inplace_one(const NPUArray& array, const Key& key);
    Handle build_binary_add(
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output,
        const Key& key);
    Handle build_binary_mul(
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output,
        const Key& key);
    Handle build_reduce_sum(
        const NPUArray& input,
        const NPUArray& output,
        const std::vector<int64_t>& axes,
        bool keepdims,
        const Key& key);
    static Key make_inplace_unary_key(const NPUArray& array, OpKind op_kind);
    static Key make_binary_key(
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output,
        OpKind op_kind);
    static Key make_reduce_sum_key(
        const NPUArray& input,
        const NPUArray& output,
        const std::vector<int64_t>& axes,
        bool keepdims);
    static bool rebind_inplace_unary(const Entry& entry, const NPUArray& array) noexcept;
    static bool rebind_binary(
        const Entry& entry,
        const NPUArray& input1,
        const NPUArray& input2,
        const NPUArray& output) noexcept;
    static bool rebind_reduce_sum(
        const Entry& entry,
        const NPUArray& input,
        const NPUArray& output) noexcept;

    mutable std::mutex mutex_;
    std::unordered_map<Key, Entry, KeyHash> entries_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    size_t builds_ = 0;
    size_t repeatable_failures_ = 0;
    size_t rebind_failures_ = 0;
};

} // namespace asnumpy::utils
