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

#include <asnumpy/utils/executor_cache.hpp>

#include <acl/acl.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_ones.h>
#include <aclnnop/aclnn_zero.h>

#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <utility>

#include <asnumpy/utils/status_handler.hpp>

namespace {

template <typename T>
inline void hash_combine(size_t& seed, const T& value) noexcept {
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

inline size_t hash_vector(const std::vector<int64_t>& values) noexcept {
    size_t seed = 0;
    for (const auto value : values) {
        hash_combine(seed, value);
    }
    return seed;
}

inline void hash_tensor_meta(size_t& seed, const asnumpy::utils::TensorDescriptorMeta& meta) noexcept {
    hash_combine(seed, hash_vector(meta.view_dims));
    hash_combine(seed, hash_vector(meta.strides));
    hash_combine(seed, static_cast<int>(meta.data_type));
    hash_combine(seed, static_cast<int>(meta.format));
    hash_combine(seed, meta.offset);
    hash_combine(seed, hash_vector(meta.storage_dims));
}

struct RebindSlot {
    enum class Kind : uint8_t {
        Input,
        Output,
    };

    Kind kind;
    size_t index;
    const aclTensor* tensor;
    void* addr;
};

inline bool rebind_slots(
    aclOpExecutor* executor,
    std::initializer_list<RebindSlot> slots) noexcept {
    if (!executor) {
        return false;
    }

    for (const auto& slot : slots) {
        if (!slot.tensor) {
            return false;
        }

        const auto status = slot.kind == RebindSlot::Kind::Input
            ? AclSetInputTensorAddr(executor, slot.index, const_cast<aclTensor*>(slot.tensor), slot.addr)
            : AclSetOutputTensorAddr(executor, slot.index, const_cast<aclTensor*>(slot.tensor), slot.addr);
        if (status != ACL_SUCCESS) {
            return false;
        }
    }

    return true;
}

} // namespace

namespace asnumpy::utils {

bool ExecutorCache::Key::operator==(const Key& other) const noexcept {
    return op_kind == other.op_kind &&
        primary_meta == other.primary_meta &&
        secondary_meta == other.secondary_meta &&
        tertiary_meta == other.tertiary_meta &&
        int_array_arg == other.int_array_arg &&
        bool_arg == other.bool_arg;
}

ExecutorCache& ExecutorCache::instance() {
    static ExecutorCache cache;
    return cache;
}

size_t ExecutorCache::KeyHash::operator()(const Key& key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, static_cast<uint8_t>(key.op_kind));
    hash_tensor_meta(seed, key.primary_meta);
    hash_tensor_meta(seed, key.secondary_meta);
    hash_tensor_meta(seed, key.tertiary_meta);
    hash_combine(seed, hash_vector(key.int_array_arg));
    hash_combine(seed, key.bool_arg);
    return seed;
}

ExecutorCache::Key ExecutorCache::make_inplace_unary_key(const NPUArray& array, OpKind op_kind) {
    Key key;
    key.op_kind = op_kind;
    key.primary_meta = {
        array.shape,
        array.strides,
        array.aclDtype,
        ACL_FORMAT_ND,
        0,
        array.shape,
    };
    return key;
}

ExecutorCache::Key ExecutorCache::make_binary_key(
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output,
    OpKind op_kind) {
    Key key;
    key.op_kind = op_kind;
    key.primary_meta = {
        input1.shape,
        input1.strides,
        input1.aclDtype,
        ACL_FORMAT_ND,
        0,
        input1.shape,
    };
    key.secondary_meta = {
        input2.shape,
        input2.strides,
        input2.aclDtype,
        ACL_FORMAT_ND,
        0,
        input2.shape,
    };
    key.tertiary_meta = {
        output.shape,
        output.strides,
        output.aclDtype,
        ACL_FORMAT_ND,
        0,
        output.shape,
    };
    return key;
}

ExecutorCache::Key ExecutorCache::make_reduce_sum_key(
    const NPUArray& input,
    const NPUArray& output,
    const std::vector<int64_t>& axes,
    bool keepdims) {
    Key key;
    key.op_kind = OpKind::ReduceSum;
    key.primary_meta = {
        input.shape,
        input.strides,
        input.aclDtype,
        ACL_FORMAT_ND,
        0,
        input.shape,
    };
    key.secondary_meta = {
        output.shape,
        output.strides,
        output.aclDtype,
        ACL_FORMAT_ND,
        0,
        output.shape,
    };
    key.int_array_arg = axes;
    key.bool_arg = keepdims;
    return key;
}

bool ExecutorCache::rebind_inplace_unary(const Entry& entry, const NPUArray& array) noexcept {
    return rebind_slots(entry.executor, {
        {RebindSlot::Kind::Output, 0, array.tensorPtr, array.device_address()},
        {RebindSlot::Kind::Input, 0, array.tensorPtr, array.device_address()},
    });
}

bool ExecutorCache::rebind_binary(
    const Entry& entry,
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output) noexcept {
    return rebind_slots(entry.executor, {
        {RebindSlot::Kind::Input, 0, input1.tensorPtr, input1.device_address()},
        {RebindSlot::Kind::Input, 1, input2.tensorPtr, input2.device_address()},
        {RebindSlot::Kind::Output, 0, output.tensorPtr, output.device_address()},
    });
}

bool ExecutorCache::rebind_reduce_sum(
    const Entry& entry,
    const NPUArray& input,
    const NPUArray& output) noexcept {
    return rebind_slots(entry.executor, {
        {RebindSlot::Kind::Input, 0, input.tensorPtr, input.device_address()},
        {RebindSlot::Kind::Output, 0, output.tensorPtr, output.device_address()},
    });
}

ExecutorCache::Handle ExecutorCache::build_inplace_zero(const NPUArray& array, const Key& key) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnInplaceZeroGetWorkspaceSize(array.tensorPtr, &workspace_size, &executor);
    ACLNN_CHECK(status, "aclnnInplaceZeroGetWorkspaceSize");

    const auto repeatable_status = aclSetAclOpExecutorRepeatable(executor);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++misses_;
        ++builds_;
        if (repeatable_status == ACL_SUCCESS) {
            entries_[key] = {executor, workspace_size, {}};
        } else {
            ++repeatable_failures_;
        }
    }

    return {executor, workspace_size, {}};
}

ExecutorCache::Handle ExecutorCache::build_inplace_one(const NPUArray& array, const Key& key) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnInplaceOneGetWorkspaceSize(array.tensorPtr, &workspace_size, &executor);
    ACLNN_CHECK(status, "aclnnInplaceOneGetWorkspaceSize");

    const auto repeatable_status = aclSetAclOpExecutorRepeatable(executor);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++misses_;
        ++builds_;
        if (repeatable_status == ACL_SUCCESS) {
            entries_[key] = {executor, workspace_size, {}};
        } else {
            ++repeatable_failures_;
        }
    }

    return {executor, workspace_size, {}};
}

ExecutorCache::Handle ExecutorCache::build_binary_add(
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output,
    const Key& key) {
    int32_t one = 1;
    aclScalar* alpha_scalar = aclCreateScalar(&one, ACL_INT32);
    if (!alpha_scalar) {
        throw std::runtime_error("aclCreateScalar failed for add alpha");
    }

    std::shared_ptr<void> retained_alpha(
        alpha_scalar,
        [](void* ptr) {
            if (ptr != nullptr) {
                aclDestroyScalar(static_cast<aclScalar*>(ptr));
            }
        });

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnAddGetWorkspaceSize(
        input1.tensorPtr,
        input2.tensorPtr,
        static_cast<aclScalar*>(retained_alpha.get()),
        output.tensorPtr,
        &workspace_size,
        &executor);
    ACLNN_CHECK(status, "aclnnAddGetWorkspaceSize");

    std::vector<std::shared_ptr<void>> retained_args{retained_alpha};
    const auto repeatable_status = aclSetAclOpExecutorRepeatable(executor);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++misses_;
        ++builds_;
        if (repeatable_status == ACL_SUCCESS) {
            entries_[key] = {executor, workspace_size, retained_args};
        } else {
            ++repeatable_failures_;
        }
    }

    return {executor, workspace_size, retained_args};
}

ExecutorCache::Handle ExecutorCache::build_binary_mul(
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output,
    const Key& key) {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnMulGetWorkspaceSize(
        input1.tensorPtr,
        input2.tensorPtr,
        output.tensorPtr,
        &workspace_size,
        &executor);
    ACLNN_CHECK(status, "aclnnMulGetWorkspaceSize");

    const auto repeatable_status = aclSetAclOpExecutorRepeatable(executor);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++misses_;
        ++builds_;
        if (repeatable_status == ACL_SUCCESS) {
            entries_[key] = {executor, workspace_size, {}};
        } else {
            ++repeatable_failures_;
        }
    }

    return {executor, workspace_size, {}};
}

ExecutorCache::Handle ExecutorCache::build_reduce_sum(
    const NPUArray& input,
    const NPUArray& output,
    const std::vector<int64_t>& axes,
    bool keepdims,
    const Key& key) {
    aclIntArray* axis_array = aclCreateIntArray(axes.data(), axes.size());
    if (!axis_array) {
        throw std::runtime_error("aclCreateIntArray failed for reduce_sum axes");
    }
    std::shared_ptr<void> retained_axis(
        axis_array,
        [](void* ptr) {
            if (ptr != nullptr) {
                aclDestroyIntArray(static_cast<aclIntArray*>(ptr));
            }
        });

    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnReduceSumGetWorkspaceSize(
        input.tensorPtr,
        static_cast<aclIntArray*>(retained_axis.get()),
        keepdims,
        output.aclDtype,
        output.tensorPtr,
        &workspace_size,
        &executor);
    ACLNN_CHECK(status, "aclnnReduceSumGetWorkspaceSize");

    const auto repeatable_status = aclSetAclOpExecutorRepeatable(executor);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++misses_;
        ++builds_;
        if (repeatable_status == ACL_SUCCESS) {
            entries_[key] = {executor, workspace_size, {retained_axis}};
        } else {
            ++repeatable_failures_;
        }
    }

    return {executor, workspace_size, {retained_axis}};
}

ExecutorCache::Handle ExecutorCache::prepare_inplace_zero(const NPUArray& array) {
    const auto key = make_inplace_unary_key(array, OpKind::InplaceZero);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const auto entry = iter->second;
            if (rebind_inplace_unary(entry, array)) {
                ++hits_;
                return {entry.executor, entry.workspace_size, entry.retained_args};
            }
            ++rebind_failures_;
            entries_.erase(iter);
        }
    }

    return build_inplace_zero(array, key);
}

ExecutorCache::Handle ExecutorCache::prepare_inplace_one(const NPUArray& array) {
    const auto key = make_inplace_unary_key(array, OpKind::InplaceOne);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const auto entry = iter->second;
            if (rebind_inplace_unary(entry, array)) {
                ++hits_;
                return {entry.executor, entry.workspace_size, entry.retained_args};
            }
            ++rebind_failures_;
            entries_.erase(iter);
        }
    }

    return build_inplace_one(array, key);
}

ExecutorCache::Handle ExecutorCache::prepare_binary_add(
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output) {
    const auto key = make_binary_key(input1, input2, output, OpKind::BinaryAdd);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const auto entry = iter->second;
            if (rebind_binary(entry, input1, input2, output)) {
                ++hits_;
                return {entry.executor, entry.workspace_size, entry.retained_args};
            }
            ++rebind_failures_;
            entries_.erase(iter);
        }
    }

    return build_binary_add(input1, input2, output, key);
}

ExecutorCache::Handle ExecutorCache::prepare_binary_mul(
    const NPUArray& input1,
    const NPUArray& input2,
    const NPUArray& output) {
    const auto key = make_binary_key(input1, input2, output, OpKind::BinaryMul);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const auto entry = iter->second;
            if (rebind_binary(entry, input1, input2, output)) {
                ++hits_;
                return {entry.executor, entry.workspace_size, entry.retained_args};
            }
            ++rebind_failures_;
            entries_.erase(iter);
        }
    }

    return build_binary_mul(input1, input2, output, key);
}

ExecutorCache::Handle ExecutorCache::prepare_reduce_sum(
    const NPUArray& input,
    const NPUArray& output,
    const std::vector<int64_t>& axes,
    bool keepdims) {
    const auto key = make_reduce_sum_key(input, output, axes, keepdims);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto iter = entries_.find(key);
        if (iter != entries_.end()) {
            const auto entry = iter->second;
            if (rebind_reduce_sum(entry, input, output)) {
                ++hits_;
                return {entry.executor, entry.workspace_size, entry.retained_args};
            }
            ++rebind_failures_;
            entries_.erase(iter);
        }
    }

    return build_reduce_sum(input, output, axes, keepdims, key);
}

void ExecutorCache::clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    // This POC drops reusable entries only. The current codebase relies on
    // CANN-managed executor lifetime after GetWorkspaceSize and does not
    // explicitly destroy aclOpExecutor instances.
    entries_.clear();
}

void ExecutorCache::reset_stats() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    builds_ = 0;
    repeatable_failures_ = 0;
    rebind_failures_ = 0;
}

ExecutorCacheStats ExecutorCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ExecutorCacheStats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.builds = builds_;
    stats.repeatable_failures = repeatable_failures_;
    stats.rebind_failures = rebind_failures_;
    stats.cached_count = entries_.size();
    return stats;
}

} // namespace asnumpy::utils
