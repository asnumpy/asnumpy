#pragma once

#include <acl/acl.h>
#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "asnumpy/memory/MemoryPool.hpp"

namespace asnumpy {
namespace memory {

class VirtualMemoryManager {
public:
    struct UnmapResult {
        aclError status = ACL_SUCCESS;
        size_t unmapped_chunk_count = 0;
    };

    struct PhysicalAllocation {
        void* handle = nullptr;
        size_t size = 0;
    };

    struct Reservation {
        void* base = nullptr;
        size_t size = 0;
    };

    static VirtualMemoryManager& instance();

    VirtualMemoryManager(const VirtualMemoryManager&) = delete;
    VirtualMemoryManager& operator=(const VirtualMemoryManager&) = delete;

    bool enabled(PoolDomain domain) const;
    bool available() const;
    size_t chunk_bytes(PoolDomain domain);
    std::string last_error() const;
    bool should_report_fallback_once() const;

    std::shared_ptr<PhysicalAllocation> allocate_physical(PoolDomain domain, size_t size);
    void free_physical(const std::shared_ptr<PhysicalAllocation>& allocation);

    std::shared_ptr<Reservation> reserve(size_t size);
    aclError release_reservation(const std::shared_ptr<Reservation>& reservation);

    void map(const std::shared_ptr<Reservation>& reservation,
             size_t offset,
             const std::shared_ptr<PhysicalAllocation>& allocation);
    UnmapResult unmap(const std::shared_ptr<Reservation>& reservation, size_t offset, size_t size);

private:
    VirtualMemoryManager() = default;

    bool probe_once() const;
    bool env_enabled(PoolDomain domain) const;
    size_t query_chunk_bytes(PoolDomain domain) const;
    void set_last_error(const std::string& message) const;

    mutable bool probed_ = false;
    mutable bool available_ = false;
    mutable std::string last_error_;
    mutable size_t probed_chunk_bytes_ = 0;
    mutable std::atomic<bool> fallback_reported_{false};
};

} // namespace memory
} // namespace asnumpy
