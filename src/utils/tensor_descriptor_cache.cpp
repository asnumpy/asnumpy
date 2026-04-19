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

#include <asnumpy/utils/tensor_descriptor_cache.hpp>
#include <fmt/format.h>
#include <stdexcept>
#include <utility>

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

} // namespace

namespace asnumpy::utils {

bool TensorDescriptorMeta::operator==(const TensorDescriptorMeta& other) const noexcept {
    return view_dims == other.view_dims &&
           strides == other.strides &&
           data_type == other.data_type &&
           format == other.format &&
           offset == other.offset &&
           storage_dims == other.storage_dims;
}

TensorDescriptorCache& TensorDescriptorCache::instance() {
    static TensorDescriptorCache cache;
    return cache;
}

TensorDescriptorCache::~TensorDescriptorCache() noexcept {
    clear();
}

size_t TensorDescriptorCache::TensorDescriptorMetaHash::operator()(const TensorDescriptorMeta& meta) const noexcept {
    size_t seed = 0;
    hash_combine(seed, hash_vector(meta.view_dims));
    hash_combine(seed, hash_vector(meta.strides));
    hash_combine(seed, static_cast<int>(meta.data_type));
    hash_combine(seed, static_cast<int>(meta.format));
    hash_combine(seed, meta.offset);
    hash_combine(seed, hash_vector(meta.storage_dims));
    return seed;
}

aclTensor* TensorDescriptorCache::create_tensor(const TensorDescriptorMeta& meta, void* device_ptr) {
    auto* tensor = aclCreateTensor(
        meta.view_dims.data(),
        meta.view_dims.size(),
        meta.data_type,
        meta.strides.data(),
        meta.offset,
        meta.format,
        meta.storage_dims.data(),
        meta.storage_dims.size(),
        device_ptr
    );
    if (!tensor) {
        throw std::runtime_error("aclCreateTensor failed in TensorDescriptorCache");
    }
    return tensor;
}

aclTensor* TensorDescriptorCache::acquire(const TensorDescriptorMeta& meta, void* device_ptr) {
    aclTensor* tensor = nullptr;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = freelists_.find(meta);
        if (iter != freelists_.end() && !iter->second.empty()) {
            tensor = iter->second.back();
            iter->second.pop_back();
            --cached_count_;
            if (iter->second.empty()) {
                freelists_.erase(iter);
            }
            ++hits_;
        } else {
            ++misses_;
        }
    }

    if (tensor) {
        const auto status = aclSetRawTensorAddr(tensor, device_ptr);
        if (status != ACL_SUCCESS) {
            auto destroy_status = aclDestroyTensor(tensor);
            (void)destroy_status;
            std::lock_guard<std::mutex> lock(mutex_);
            ++destroyed_;
            throw std::runtime_error(fmt::format("aclSetRawTensorAddr failed in TensorDescriptorCache. error code: {}", status));
        }
        return tensor;
    }

    tensor = create_tensor(meta, device_ptr);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++created_;
    }
    return tensor;
}

void TensorDescriptorCache::release(const TensorDescriptorMeta& meta, aclTensor* tensor) noexcept {
    if (!tensor) {
        return;
    }

    bool destroy_tensor = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = freelists_.find(meta);
        if (cached_count_ >= kMaxCachedDescriptors) {
            destroy_tensor = true;
        } else {
            if (iter == freelists_.end()) {
                iter = freelists_.emplace(meta, std::vector<aclTensor*>{}).first;
            }
            if (iter->second.size() >= kMaxFreeListSize) {
                destroy_tensor = true;
                if (iter->second.empty()) {
                    freelists_.erase(iter);
                }
            } else {
                iter->second.push_back(tensor);
                ++cached_count_;
            }
        }

        if (destroy_tensor) {
            ++destroyed_;
            ++evicted_;
        }
    }

    if (destroy_tensor) {
        auto status = aclDestroyTensor(tensor);
        (void)status;
    }
}

void TensorDescriptorCache::clear() noexcept {
    FreeListMap freelists_to_destroy;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        freelists_to_destroy.swap(freelists_);
        cached_count_ = 0;
    }

    size_t destroyed_now = 0;
    for (auto& [meta, tensors] : freelists_to_destroy) {
        (void)meta;
        for (auto* tensor : tensors) {
            if (tensor) {
                auto status = aclDestroyTensor(tensor);
                (void)status;
                ++destroyed_now;
            }
        }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    destroyed_ += destroyed_now;
}

void TensorDescriptorCache::reset_stats() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    hits_ = 0;
    misses_ = 0;
    created_ = 0;
    destroyed_ = 0;
    evicted_ = 0;
}

TensorDescriptorCacheStats TensorDescriptorCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    TensorDescriptorCacheStats stats;
    stats.hits = hits_;
    stats.misses = misses_;
    stats.created = created_;
    stats.destroyed = destroyed_;
    stats.evicted = evicted_;
    stats.cached_count = cached_count_;
    stats.freelist_keys = freelists_.size();
    return stats;
}

} // namespace asnumpy::utils
