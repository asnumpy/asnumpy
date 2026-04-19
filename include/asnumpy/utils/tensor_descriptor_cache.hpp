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

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace asnumpy::utils {

struct TensorDescriptorMeta {
    std::vector<int64_t> view_dims;
    std::vector<int64_t> strides;
    aclDataType data_type = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_ND;
    int64_t offset = 0;
    std::vector<int64_t> storage_dims;

    bool operator==(const TensorDescriptorMeta& other) const noexcept;
};

struct TensorDescriptorCacheStats {
    size_t hits = 0;
    size_t misses = 0;
    size_t created = 0;
    size_t destroyed = 0;
    size_t evicted = 0;
    size_t cached_count = 0;
    size_t freelist_keys = 0;
};

class TensorDescriptorCache {
public:
    static TensorDescriptorCache& instance();
    ~TensorDescriptorCache() noexcept;

    aclTensor* acquire(const TensorDescriptorMeta& meta, void* device_ptr);
    void release(const TensorDescriptorMeta& meta, aclTensor* tensor) noexcept;
    void clear() noexcept;
    void reset_stats() noexcept;
    TensorDescriptorCacheStats stats() const;

private:
    static constexpr size_t kMaxFreeListSize = 32;
    static constexpr size_t kMaxCachedDescriptors = 4096;

    TensorDescriptorCache() = default;

    struct TensorDescriptorMetaHash {
        size_t operator()(const TensorDescriptorMeta& meta) const noexcept;
    };

    using FreeListMap = std::unordered_map<TensorDescriptorMeta, std::vector<aclTensor*>, TensorDescriptorMetaHash>;

    static aclTensor* create_tensor(const TensorDescriptorMeta& meta, void* device_ptr);

    mutable std::mutex mutex_;
    FreeListMap freelists_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    size_t created_ = 0;
    size_t destroyed_ = 0;
    size_t evicted_ = 0;
    size_t cached_count_ = 0;
};

} // namespace asnumpy::utils
