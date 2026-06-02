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

#include <pybind11/pybind11.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <asnumpy/memory/MemoryPool.hpp>
#include <asnumpy/utils/executor_cache.hpp>
#include <asnumpy/utils/tensor_descriptor_cache.hpp>

void bind_memory(pybind11::module_& m) {
    auto parse_domain = [](const pybind11::object& domain_obj) -> std::optional<asnumpy::memory::PoolDomain> {
        if (domain_obj.is_none()) {
            return std::nullopt;
        }
        const auto domain = pybind11::cast<std::string>(domain_obj);
        if (domain == "tensor") {
            return asnumpy::memory::PoolDomain::Tensor;
        }
        if (domain == "workspace") {
            return asnumpy::memory::PoolDomain::Workspace;
        }
        throw std::runtime_error("memory domain must be 'tensor', 'workspace', or None");
    };

    auto stats_to_dict = [](const asnumpy::memory::MemoryPoolStats& stats, bool verbose) {
        pybind11::dict result;
        result["cache_hits"] = pybind11::int_(stats.cache_hits);
        result["cache_misses"] = pybind11::int_(stats.cache_misses);
        result["system_allocations"] = pybind11::int_(stats.system_allocations);
        result["system_frees"] = pybind11::int_(stats.system_frees);
        result["active_bytes"] = pybind11::int_(stats.active_bytes);
        result["cached_bytes"] = pybind11::int_(stats.cached_bytes);
        result["largest_free_run_bytes"] = pybind11::int_(stats.largest_free_run_bytes);
        result["largest_free_block_bytes"] = pybind11::int_(stats.largest_free_block_bytes);
        result["internal_fragmentation_bytes"] = pybind11::int_(stats.internal_fragmentation_bytes);
        result["vmm_internal_fragmentation_bytes"] = pybind11::int_(stats.vmm_internal_fragmentation_bytes);
        result["external_fragmentation_bytes"] = pybind11::int_(stats.external_fragmentation_bytes);
        result["hot_retained_empty_runs"] = pybind11::int_(stats.hot_retained_empty_runs);
        result["stitched_reuse_hits"] = pybind11::int_(stats.stitched_reuse_hits);
        result["vmm_fallback_allocations"] = pybind11::int_(stats.vmm_fallback_allocations);
        result["chunk_cache_bytes"] = pybind11::int_(stats.chunk_cache_bytes);
        result["chunk_cache_count"] = pybind11::int_(stats.chunk_cache_count);
        result["stitched_segment_count"] = pybind11::int_(stats.stitched_segment_count);
        result["internal_fragmentation_ratio_pct"] = pybind11::float_(stats.internal_fragmentation_ratio_pct);
        result["external_fragmentation_ratio_pct"] = pybind11::float_(stats.external_fragmentation_ratio_pct);
        result["run_utilization_pct"] = pybind11::float_(stats.run_utilization_pct);
        if (!verbose) {
            return result;
        }

        result["allocation_requests"] = pybind11::int_(stats.allocation_requests);
        result["free_requests"] = pybind11::int_(stats.free_requests);
        result["small_allocation_requests"] = pybind11::int_(stats.small_allocation_requests);
        result["small_cache_hits"] = pybind11::int_(stats.small_cache_hits);
        result["small_cache_misses"] = pybind11::int_(stats.small_cache_misses);
        result["large_allocation_requests"] = pybind11::int_(stats.large_allocation_requests);
        result["large_cache_hits"] = pybind11::int_(stats.large_cache_hits);
        result["large_cache_misses"] = pybind11::int_(stats.large_cache_misses);
        result["active_requested_bytes"] = pybind11::int_(stats.active_requested_bytes);
        result["system_bytes"] = pybind11::int_(stats.system_bytes);
        result["peak_active_bytes"] = pybind11::int_(stats.peak_active_bytes);
        result["small_run_count"] = pybind11::int_(stats.small_run_count);
        result["large_arena_count"] = pybind11::int_(stats.large_arena_count);
        result["empty_small_run_count"] = pybind11::int_(stats.empty_small_run_count);
        result["partial_small_run_count"] = pybind11::int_(stats.partial_small_run_count);
        result["full_small_run_count"] = pybind11::int_(stats.full_small_run_count);
        result["small_slots_total"] = pybind11::int_(stats.small_slots_total);
        result["small_slots_free"] = pybind11::int_(stats.small_slots_free);
        result["bitmap_words_scanned"] = pybind11::int_(stats.bitmap_words_scanned);
        result["summary_words_scanned"] = pybind11::int_(stats.summary_words_scanned);
        result["dirty_large_block_count"] = pybind11::int_(stats.dirty_large_block_count);
        result["retained_large_block_count"] = pybind11::int_(stats.retained_large_block_count);
        result["dirty_large_bytes"] = pybind11::int_(stats.dirty_large_bytes);
        result["retained_large_bytes"] = pybind11::int_(stats.retained_large_bytes);
        result["large_split_count"] = pybind11::int_(stats.large_split_count);
        result["large_merge_count"] = pybind11::int_(stats.large_merge_count);
        result["large_decay_count"] = pybind11::int_(stats.large_decay_count);
        result["trimmed_bytes"] = pybind11::int_(stats.trimmed_bytes);
        result["small_free_bytes"] = pybind11::int_(stats.small_free_bytes);
        result["small_external_fragmentation_bytes"] = pybind11::int_(stats.small_external_fragmentation_bytes);
        result["large_external_fragmentation_bytes"] = pybind11::int_(stats.large_external_fragmentation_bytes);
        result["small_external_fragmentation_ratio_pct"] = pybind11::float_(stats.small_external_fragmentation_ratio_pct);
        result["large_external_fragmentation_ratio_pct"] = pybind11::float_(stats.large_external_fragmentation_ratio_pct);
        return result;
    };

    auto descriptor_stats_to_dict = [](const asnumpy::utils::TensorDescriptorCacheStats& stats) {
        pybind11::dict result;
        result["hits"] = pybind11::int_(stats.hits);
        result["misses"] = pybind11::int_(stats.misses);
        result["created"] = pybind11::int_(stats.created);
        result["destroyed"] = pybind11::int_(stats.destroyed);
        result["evicted"] = pybind11::int_(stats.evicted);
        result["cached_count"] = pybind11::int_(stats.cached_count);
        result["freelist_keys"] = pybind11::int_(stats.freelist_keys);
        return result;
    };

    auto executor_stats_to_dict = [](const asnumpy::utils::ExecutorCacheStats& stats) {
        pybind11::dict result;
        result["hits"] = pybind11::int_(stats.hits);
        result["misses"] = pybind11::int_(stats.misses);
        result["builds"] = pybind11::int_(stats.builds);
        result["repeatable_failures"] = pybind11::int_(stats.repeatable_failures);
        result["rebind_failures"] = pybind11::int_(stats.rebind_failures);
        result["cached_count"] = pybind11::int_(stats.cached_count);
        return result;
    };

    m.def("clear_cache", [parse_domain](pybind11::object domain_obj) {
        auto domain = parse_domain(domain_obj);
        if (domain.has_value()) {
            asnumpy::memory::MemoryPool::instance().clear_cache(*domain);
        } else {
            asnumpy::memory::MemoryPool::instance().clear_cache();
        }
    }, pybind11::arg("domain") = pybind11::none(),
       "Clear all free memory cached by the memory pool");

    m.def("trim", [parse_domain](pybind11::object domain_obj) {
        auto domain = parse_domain(domain_obj);
        if (domain.has_value()) {
            asnumpy::memory::MemoryPool::instance().trim(*domain);
        } else {
            asnumpy::memory::MemoryPool::instance().trim();
        }
    }, pybind11::arg("domain") = pybind11::none(),
       "Release fully free runs and arenas back to the runtime");

    m.def("refresh_config", []() {
        asnumpy::memory::MemoryPool::instance().refresh_config();
    }, "Refresh memory pool configuration from environment variables");

    m.def("reset_stats", [parse_domain](pybind11::object domain_obj) {
        auto domain = parse_domain(domain_obj);
        if (domain.has_value()) {
            asnumpy::memory::MemoryPool::instance().reset_stats(*domain);
        } else {
            asnumpy::memory::MemoryPool::instance().reset_stats();
        }
    }, pybind11::arg("domain") = pybind11::none(),
       "Reset cumulative memory pool counters while preserving current pool state");

    m.def("stats", [parse_domain, stats_to_dict](pybind11::object domain_obj, bool verbose) {
        auto domain = parse_domain(domain_obj);
        if (domain.has_value()) {
            return stats_to_dict(asnumpy::memory::MemoryPool::instance().stats(*domain), verbose);
        }
        return stats_to_dict(asnumpy::memory::MemoryPool::instance().stats(), verbose);
    }, pybind11::arg("domain") = pybind11::none(),
       pybind11::arg("verbose") = false,
       "Return current memory pool counters");

    m.def("debug_stats", [parse_domain, stats_to_dict](pybind11::object domain_obj) {
        auto domain = parse_domain(domain_obj);
        if (domain.has_value()) {
            return stats_to_dict(asnumpy::memory::MemoryPool::instance().stats(*domain), true);
        }
        return stats_to_dict(asnumpy::memory::MemoryPool::instance().stats(), true);
    }, pybind11::arg("domain") = pybind11::none(),
       "Return full memory pool diagnostic counters");

    m.def("descriptor_stats", [descriptor_stats_to_dict]() {
        return descriptor_stats_to_dict(asnumpy::utils::TensorDescriptorCache::instance().stats());
    }, "Return tensor descriptor cache counters");

    m.def("clear_descriptor_cache", []() {
        asnumpy::utils::TensorDescriptorCache::instance().clear();
    }, "Destroy all cached tensor descriptors");

    m.def("reset_descriptor_stats", []() {
        asnumpy::utils::TensorDescriptorCache::instance().reset_stats();
    }, "Reset tensor descriptor cache counters");

    m.def("executor_stats", [executor_stats_to_dict]() {
        return executor_stats_to_dict(asnumpy::utils::ExecutorCache::instance().stats());
    }, "Return repeatable executor cache counters");

    m.def("clear_executor_cache", []() {
        asnumpy::utils::ExecutorCache::instance().clear();
    }, "Drop cached repeatable executors");

    m.def("reset_executor_stats", []() {
        asnumpy::utils::ExecutorCache::instance().reset_stats();
    }, "Reset repeatable executor cache counters");

    m.def("shutdown_runtime", []() {
        asnumpy::utils::ExecutorCache::instance().clear();
        asnumpy::utils::TensorDescriptorCache::instance().clear();
        asnumpy::memory::MemoryPool::instance().clear_cache();
    }, "Clear runtime reuse layers in dependency order: executor -> descriptor -> memory");

    m.def("benchmark_allocator", [parse_domain, stats_to_dict](
        size_t size_bytes,
        size_t iterations,
        size_t warmup,
        pybind11::object domain_obj
    ) {
        using clock = std::chrono::high_resolution_clock;

        auto domain = parse_domain(domain_obj).value_or(asnumpy::memory::PoolDomain::Tensor);
        auto& pool = asnumpy::memory::MemoryPool::instance();

        auto run_baseline = [size_bytes](size_t count) {
            for (size_t i = 0; i < count; ++i) {
                void* ptr = nullptr;
                auto ret = aclrtMalloc(&ptr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("allocator benchmark baseline aclrtMalloc failed");
                }
                ret = aclrtFree(ptr);
                if (ret != ACL_SUCCESS) {
                    throw std::runtime_error("allocator benchmark baseline aclrtFree failed");
                }
            }
        };

        auto run_pool = [&pool, size_bytes, domain](size_t count) {
            for (size_t i = 0; i < count; ++i) {
                void* ptr = pool.malloc(size_bytes, domain);
                pool.free(ptr, domain);
            }
        };

        run_baseline(warmup);
        auto baseline_start = clock::now();
        run_baseline(iterations);
        auto baseline_end = clock::now();

        pool.clear_cache(domain);
        pool.reset_stats(domain);
        run_pool(warmup);
        pool.reset_stats(domain);
        auto pooled_start = clock::now();
        run_pool(iterations);
        auto pooled_end = clock::now();
        const auto pooled_stats = pool.stats(domain);

        const auto baseline_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            baseline_end - baseline_start
        ).count();
        const auto pooled_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            pooled_end - pooled_start
        ).count();

        const double baseline_seconds = static_cast<double>(baseline_elapsed_us) / 1'000'000.0;
        const double pooled_seconds = static_cast<double>(pooled_elapsed_us) / 1'000'000.0;
        const double baseline_ops = baseline_seconds > 0.0 ? static_cast<double>(iterations) / baseline_seconds : 0.0;
        const double pooled_ops = pooled_seconds > 0.0 ? static_cast<double>(iterations) / pooled_seconds : 0.0;
        const double baseline_latency_us = iterations > 0 ? static_cast<double>(baseline_elapsed_us) / static_cast<double>(iterations) : 0.0;
        const double pooled_latency_us = iterations > 0 ? static_cast<double>(pooled_elapsed_us) / static_cast<double>(iterations) : 0.0;

        pybind11::dict result;
        result["size_bytes"] = pybind11::int_(size_bytes);
        result["iterations"] = pybind11::int_(iterations);
        result["warmup"] = pybind11::int_(warmup);
        result["baseline_ops_per_sec"] = pybind11::float_(baseline_ops);
        result["pooled_ops_per_sec"] = pybind11::float_(pooled_ops);
        result["baseline_latency_us"] = pybind11::float_(baseline_latency_us);
        result["pooled_latency_us"] = pybind11::float_(pooled_latency_us);
        result["speedup"] = pybind11::float_(baseline_ops > 0.0 ? pooled_ops / baseline_ops : 0.0);
        result["latency_delta_pct"] = pybind11::float_(
            baseline_latency_us > 0.0 ? ((pooled_latency_us - baseline_latency_us) / baseline_latency_us) * 100.0 : 0.0
        );
        result["pool_stats"] = stats_to_dict(pooled_stats, false);
        return result;
    }, pybind11::arg("size_bytes"), pybind11::arg("iterations"), pybind11::arg("warmup") = 100,
       pybind11::arg("domain") = pybind11::none(),
       "Benchmark allocator-only baseline vs pooled allocation/free cost");
}
