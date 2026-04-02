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
#include <asnumpy/memory/MemoryPool.hpp>

void bind_memory(pybind11::module_& m) {
    m.def("clear_cache", []() {
        asnumpy::memory::MemoryPool::instance().clear_cache();
    }, "Clear all free memory cached by the memory pool");

    m.def("trim", []() {
        asnumpy::memory::MemoryPool::instance().clear_cache();
    }, "Release fully free runs and arenas back to the runtime");

    m.def("stats", []() {
        const auto stats = asnumpy::memory::MemoryPool::instance().stats();
        pybind11::dict result;
        result["allocation_requests"] = pybind11::int_(stats.allocation_requests);
        result["free_requests"] = pybind11::int_(stats.free_requests);
        result["cache_hits"] = pybind11::int_(stats.cache_hits);
        result["cache_misses"] = pybind11::int_(stats.cache_misses);
        result["system_allocations"] = pybind11::int_(stats.system_allocations);
        result["system_frees"] = pybind11::int_(stats.system_frees);
        result["active_bytes"] = pybind11::int_(stats.active_bytes);
        result["cached_bytes"] = pybind11::int_(stats.cached_bytes);
        result["system_bytes"] = pybind11::int_(stats.system_bytes);
        result["peak_active_bytes"] = pybind11::int_(stats.peak_active_bytes);
        result["small_run_count"] = pybind11::int_(stats.small_run_count);
        result["large_arena_count"] = pybind11::int_(stats.large_arena_count);
        return result;
    }, "Return current memory pool counters");
}