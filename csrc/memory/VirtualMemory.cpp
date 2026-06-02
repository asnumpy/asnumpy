#include "asnumpy/memory/VirtualMemory.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace asnumpy {
namespace memory {

namespace {

constexpr size_t kFallbackChunkBytes = 2 * 1024 * 1024;
constexpr uint64_t kReserveHugePageFlag = 1;
constexpr uint32_t kMemAccessProtReadWrite = 0x3;
constexpr uint32_t kMemLocationTypeDevice = 1;
constexpr int kFeatureNotSupport = 207000;

struct CompatMemLocation {
    uint32_t id;
    uint32_t type;
};

struct CompatMemAccessDesc {
    uint32_t flags;
    CompatMemLocation location;
    uint8_t rsv[12];
};

struct CompatPhysicalMemProp {
    uint32_t handleType;
    uint32_t allocationType;
    uint32_t memAttr;
    CompatMemLocation location;
    uint64_t reserve;
};

using CompatDrvMemHandle = void*;
using ReserveMemAddressFn = aclError (*)(void**, size_t, size_t, void*, uint64_t);
using ReleaseMemAddressFn = aclError (*)(void*);
using MallocPhysicalFn = aclError (*)(CompatDrvMemHandle*, size_t, const CompatPhysicalMemProp*, uint64_t);
using FreePhysicalFn = aclError (*)(CompatDrvMemHandle);
using MapMemFn = aclError (*)(void*, size_t, size_t, CompatDrvMemHandle, uint64_t);
using UnmapMemFn = aclError (*)(void*);
using MemGetAllocationGranularityFn = aclError (*)(CompatPhysicalMemProp*, uint32_t, size_t*);
using MemSetAccessFn = aclError (*)(void*, size_t, CompatMemAccessDesc*, size_t);

struct VmmSymbols {
    ReserveMemAddressFn reserve_mem_address = nullptr;
    ReleaseMemAddressFn release_mem_address = nullptr;
    MallocPhysicalFn malloc_physical = nullptr;
    FreePhysicalFn free_physical = nullptr;
    MapMemFn map_mem = nullptr;
    UnmapMemFn unmap_mem = nullptr;
    MemGetAllocationGranularityFn get_allocation_granularity = nullptr;
    MemSetAccessFn set_mem_access = nullptr;
};

void* load_symbol(const char* name) {
#if defined(_WIN32)
    HMODULE module = GetModuleHandleA("ascendcl.dll");
    if (module == nullptr) {
        module = LoadLibraryA("ascendcl.dll");
    }
    return module == nullptr ? nullptr : reinterpret_cast<void*>(GetProcAddress(module, name));
#else
    void* symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol != nullptr) {
        return symbol;
    }

    void* handle = dlopen("libascendcl.so", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        handle = dlopen("libascendcl.so.1", RTLD_LAZY | RTLD_LOCAL);
    }
    return handle == nullptr ? nullptr : dlsym(handle, name);
#endif
}

const VmmSymbols& symbols() {
    static const VmmSymbols loaded = []() {
        VmmSymbols vmm;
        vmm.reserve_mem_address = reinterpret_cast<ReserveMemAddressFn>(load_symbol("aclrtReserveMemAddress"));
        vmm.release_mem_address = reinterpret_cast<ReleaseMemAddressFn>(load_symbol("aclrtReleaseMemAddress"));
        vmm.malloc_physical = reinterpret_cast<MallocPhysicalFn>(load_symbol("aclrtMallocPhysical"));
        vmm.free_physical = reinterpret_cast<FreePhysicalFn>(load_symbol("aclrtFreePhysical"));
        vmm.map_mem = reinterpret_cast<MapMemFn>(load_symbol("aclrtMapMem"));
        vmm.unmap_mem = reinterpret_cast<UnmapMemFn>(load_symbol("aclrtUnmapMem"));
        vmm.get_allocation_granularity =
            reinterpret_cast<MemGetAllocationGranularityFn>(load_symbol("aclrtMemGetAllocationGranularity"));
        vmm.set_mem_access = reinterpret_cast<MemSetAccessFn>(load_symbol("aclrtMemSetAccess"));
        return vmm;
    }();
    return loaded;
}

bool env_truthy(const char* value) {
    return value != nullptr && (std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
                                std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
                                std::strcmp(value, "ON") == 0);
}

bool env_falsey(const char* value) {
    return value != nullptr && (std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
                                std::strcmp(value, "FALSE") == 0 || std::strcmp(value, "off") == 0 ||
                                std::strcmp(value, "OFF") == 0);
}

uint32_t current_device_id() {
    int32_t device = 0;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) {
        return 0;
    }
    return static_cast<uint32_t>(device);
}

CompatPhysicalMemProp default_mem_prop() {
    CompatPhysicalMemProp prop{};
    prop.handleType = 0;
    prop.allocationType = 0;
    prop.memAttr = 4;
    prop.location.id = current_device_id();
    prop.location.type = kMemLocationTypeDevice;
    prop.reserve = 0;
    return prop;
}

} // namespace

VirtualMemoryManager& VirtualMemoryManager::instance() {
    static VirtualMemoryManager instance;
    return instance;
}

bool VirtualMemoryManager::env_enabled(PoolDomain domain) const {
    const char* global = std::getenv("ASN_POOL_VMM_ENABLE");
    const char* scoped = domain == PoolDomain::Workspace
        ? std::getenv("ASN_POOL_WORKSPACE_VMM_ENABLE")
        : std::getenv("ASN_POOL_TENSOR_VMM_ENABLE");

    const char* value = scoped != nullptr ? scoped : global;
    if (value == nullptr || std::strcmp(value, "auto") == 0 || std::strcmp(value, "AUTO") == 0) {
        return true;
    }
    if (env_truthy(value)) {
        return true;
    }
    if (env_falsey(value)) {
        return false;
    }
    return true;
}

void VirtualMemoryManager::set_last_error(const std::string& message) const {
    last_error_ = message;
}

std::string VirtualMemoryManager::last_error() const {
    return last_error_;
}

bool VirtualMemoryManager::should_report_fallback_once() const {
    return !fallback_reported_.exchange(true);
}

bool VirtualMemoryManager::probe_once() const {
    if (probed_) {
        return available_;
    }

    probed_ = true;
    const auto& vmm = symbols();
    if (vmm.reserve_mem_address == nullptr || vmm.release_mem_address == nullptr ||
        vmm.malloc_physical == nullptr || vmm.free_physical == nullptr ||
        vmm.map_mem == nullptr || vmm.unmap_mem == nullptr ||
        vmm.get_allocation_granularity == nullptr || vmm.set_mem_access == nullptr) {
        set_last_error("Ascend VMM symbols are not available in the current runtime");
        available_ = false;
        return false;
    }

    size_t granularity = 0;
    auto prop = default_mem_prop();
    const aclError ret = vmm.get_allocation_granularity(&prop, 0, &granularity);
    if (ret != ACL_SUCCESS || granularity == 0) {
        std::ostringstream oss;
        oss << "aclrtMemGetAllocationGranularity failed with ret=" << ret;
        set_last_error(oss.str());
        available_ = false;
        return false;
    }

    probed_chunk_bytes_ = granularity;
    available_ = true;
    return true;
}

bool VirtualMemoryManager::available() const {
    return probe_once();
}

bool VirtualMemoryManager::enabled(PoolDomain domain) const {
    return env_enabled(domain) && probe_once();
}

size_t VirtualMemoryManager::query_chunk_bytes(PoolDomain domain) const {
    (void)domain;
    const char* override_env = std::getenv("ASN_POOL_VMM_CHUNK_BYTES");
    if (override_env != nullptr && *override_env != '\0') {
        try {
            return static_cast<size_t>(std::stoull(override_env));
        } catch (const std::exception&) {
            // Ignore invalid override and fall back to runtime probe.
        }
    }

    if (probe_once() && probed_chunk_bytes_ != 0) {
        return probed_chunk_bytes_;
    }
    return kFallbackChunkBytes;
}

size_t VirtualMemoryManager::chunk_bytes(PoolDomain domain) {
    return query_chunk_bytes(domain);
}

std::shared_ptr<VirtualMemoryManager::PhysicalAllocation> VirtualMemoryManager::allocate_physical(
    PoolDomain domain,
    size_t size) {
    if (!enabled(domain)) {
        throw std::runtime_error(last_error_.empty() ? "Ascend VMM is unavailable" : last_error_);
    }

    CompatDrvMemHandle handle = nullptr;
    auto prop = default_mem_prop();
    const aclError ret = symbols().malloc_physical(&handle, size, &prop, 0);
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << "aclrtMallocPhysical failed with ret=" << ret;
        set_last_error(oss.str());
        throw std::runtime_error(oss.str());
    }

    auto allocation = std::make_shared<PhysicalAllocation>();
    allocation->handle = handle;
    allocation->size = size;
    return allocation;
}

void VirtualMemoryManager::free_physical(const std::shared_ptr<PhysicalAllocation>& allocation) {
    if (allocation == nullptr || allocation->handle == nullptr) {
        return;
    }
    const aclError ret = symbols().free_physical(static_cast<CompatDrvMemHandle>(allocation->handle));
    allocation->handle = nullptr;
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << "aclrtFreePhysical failed with ret=" << ret;
        set_last_error(oss.str());
    }
}

std::shared_ptr<VirtualMemoryManager::Reservation> VirtualMemoryManager::reserve(size_t size) {
    if (!available()) {
        throw std::runtime_error(last_error_.empty() ? "Ascend VMM is unavailable" : last_error_);
    }

    void* base = nullptr;
    const aclError ret = symbols().reserve_mem_address(&base, size, 0, nullptr, kReserveHugePageFlag);
    if (ret != ACL_SUCCESS || base == nullptr) {
        std::ostringstream oss;
        oss << "aclrtReserveMemAddress failed with ret=" << ret;
        set_last_error(oss.str());
        throw std::runtime_error(oss.str());
    }

    auto reservation = std::make_shared<Reservation>();
    reservation->base = base;
    reservation->size = size;
    return reservation;
}

aclError VirtualMemoryManager::release_reservation(const std::shared_ptr<Reservation>& reservation) {
    if (reservation == nullptr || reservation->base == nullptr) {
        return ACL_SUCCESS;
    }
    const aclError ret = symbols().release_mem_address(reservation->base);
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << "aclrtReleaseMemAddress failed with ret=" << ret;
        set_last_error(oss.str());
        return ret;
    }
    reservation->base = nullptr;
    return ACL_SUCCESS;
}

void VirtualMemoryManager::map(
    const std::shared_ptr<Reservation>& reservation,
    size_t offset,
    const std::shared_ptr<PhysicalAllocation>& allocation) {
    if (reservation == nullptr || reservation->base == nullptr || allocation == nullptr || allocation->handle == nullptr) {
        throw std::runtime_error("VirtualMemoryManager::map received invalid reservation or physical allocation");
    }

    void* map_base = static_cast<char*>(reservation->base) + offset;
    const aclError ret = symbols().map_mem(
        map_base,
        allocation->size,
        0,
        static_cast<CompatDrvMemHandle>(allocation->handle),
        0);
    if (ret != ACL_SUCCESS) {
        std::ostringstream oss;
        oss << "aclrtMapMem failed with ret=" << ret;
        set_last_error(oss.str());
        throw std::runtime_error(oss.str());
    }

    CompatMemAccessDesc access_desc{};
    access_desc.flags = kMemAccessProtReadWrite;
    access_desc.location.id = current_device_id();
    access_desc.location.type = kMemLocationTypeDevice;
    const aclError access_ret = symbols().set_mem_access(map_base, allocation->size, &access_desc, 1);
    if (access_ret != ACL_SUCCESS) {
        if (access_ret == kFeatureNotSupport) {
            available_ = false;
            const std::string message =
                "aclrtMemSetAccess returned FEATURE_NOT_SUPPORT (ret=207000); VMM disabled for this process";
            set_last_error(message);
            (void)symbols().unmap_mem(map_base);
            throw std::runtime_error(message);
        }
        (void)symbols().unmap_mem(map_base);
        std::ostringstream oss;
        oss << "aclrtMemSetAccess failed with ret=" << access_ret;
        set_last_error(oss.str());
        throw std::runtime_error(oss.str());
    }
}

VirtualMemoryManager::UnmapResult VirtualMemoryManager::unmap(
    const std::shared_ptr<Reservation>& reservation,
    size_t offset,
    size_t size) {
    UnmapResult result;
    if (reservation == nullptr || reservation->base == nullptr || size == 0) {
        return result;
    }

    const size_t chunk = query_chunk_bytes(PoolDomain::Tensor);
    const size_t iterations = chunk == 0 ? 1 : (size / chunk);
    for (size_t index = 0; index < iterations; ++index) {
        void* map_base = static_cast<char*>(reservation->base) + offset + (index * chunk);
        const aclError ret = symbols().unmap_mem(map_base);
        if (ret != ACL_SUCCESS) {
            std::ostringstream oss;
            oss << "aclrtUnmapMem failed with ret=" << ret;
            set_last_error(oss.str());
            result.status = ret;
            return result;
        }
        ++result.unmapped_chunk_count;
    }
    return result;
}

} // namespace memory
} // namespace asnumpy
