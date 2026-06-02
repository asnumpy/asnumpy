#include "asnumpy/memory/MemoryPool.hpp"

#include "asnumpy/memory/VirtualMemory.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace asnumpy {
namespace memory {

namespace {

constexpr size_t kAlignment = 512;
constexpr size_t kSmallAllocationMax = 1 * 1024 * 1024;
constexpr size_t kDefaultSmallRunTargetSize = 2 * 1024 * 1024;
constexpr size_t kWorkspaceSmallRunTargetSize = 1 * 1024 * 1024;
constexpr size_t kSmallRunMinSlots = 4;
constexpr size_t kSmallRunMaxSlots = 128;
constexpr size_t kLargeMinSplitSize = 4 * 1024;
constexpr uint64_t kTensorLargeDirtyDecayEpochs = 4;
constexpr uint64_t kWorkspaceLargeDirtyDecayEpochs = 1;
constexpr size_t kDefaultTensorChunkCacheBytes = 256 * 1024 * 1024;
constexpr size_t kDefaultWorkspaceChunkCacheBytes = 64 * 1024 * 1024;

constexpr std::array<size_t, 20> kSmallSizeClasses = {
    512, 1024, 2048, 4096, 8192, 16384,
    32768, 49152, 65536, 98304,
    131072, 196608, 262144, 393216,
    524288, 655360, 786432, 917504,
    983040, 1048576,
};

bool pool_enabled_from_env() {
    const char* env = std::getenv("ASN_ENABLE_POOL");
    return env == nullptr || std::string(env) != "0";
}

std::optional<size_t> size_t_from_env(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return std::nullopt;
    }

    try {
        const auto value = std::stoull(env);
        if (value > std::numeric_limits<size_t>::max()) {
            throw std::out_of_range("size_t overflow");
        }
        return static_cast<size_t>(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid unsigned integer in environment variable " +
                                 std::string(name) + ": " + env);
    }
}

std::optional<uint64_t> uint64_from_env(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return std::nullopt;
    }

    try {
        return std::stoull(env);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid unsigned integer in environment variable " +
                                 std::string(name) + ": " + env);
    }
}

size_t size_t_env_or(
    const char* domain_name,
    const char* shared_name,
    size_t default_value) {
    if (auto value = size_t_from_env(domain_name)) {
        return *value;
    }
    if (shared_name != nullptr) {
        if (auto value = size_t_from_env(shared_name)) {
            return *value;
        }
    }
    return default_value;
}

uint64_t uint64_env_or(
    const char* domain_name,
    const char* shared_name,
    uint64_t default_value) {
    if (auto value = uint64_from_env(domain_name)) {
        return *value;
    }
    if (shared_name != nullptr) {
        if (auto value = uint64_from_env(shared_name)) {
            return *value;
        }
    }
    return default_value;
}

} // namespace

struct MemoryPool::PhysicalChunk {
    std::shared_ptr<VirtualMemoryManager::PhysicalAllocation> allocation;
    size_t size = 0;
    PoolDomain domain = PoolDomain::Tensor;
    uint64_t last_used_epoch = 0;
    bool mapped = false;
    bool in_chunk_cache = false;
};

struct MemoryPool::VirtualReservation {
    std::shared_ptr<VirtualMemoryManager::Reservation> handle;
    size_t size = 0;
};

struct MemoryPool::StitchedSegment {
    enum class State : uint8_t {
        Healthy,
        Quarantined,
    };

    std::shared_ptr<VirtualReservation> reservation;
    std::vector<std::shared_ptr<PhysicalChunk>> chunks;
    size_t chunk_bytes = 0;
    size_t total_bytes = 0;
    size_t mapped_chunk_count = 0;
    State state = State::Healthy;
};

MemoryPool& MemoryPool::instance() {
    static MemoryPool instance;
    return instance;
}

MemoryPool::MemoryPool() {
    refresh_config_locked();
}

MemoryPool::~MemoryPool() {
    try {
        clear_cache();
    } catch (...) {
    }
}

MemoryPool::DomainState& MemoryPool::domain_state(PoolDomain domain) {
    return domain == PoolDomain::Workspace ? workspace_domain_ : tensor_domain_;
}

const MemoryPool::DomainState& MemoryPool::domain_state(PoolDomain domain) const {
    return domain == PoolDomain::Workspace ? workspace_domain_ : tensor_domain_;
}

const MemoryPool::DomainConfig& MemoryPool::domain_config(PoolDomain domain) const {
    return domain == PoolDomain::Workspace ? workspace_config_ : tensor_config_;
}

void MemoryPool::refresh_config() {
    std::lock_guard<std::mutex> lock(mutex_);
    refresh_config_locked();
}

void MemoryPool::refresh_config_locked() {
    pool_enabled_.store(pool_enabled_from_env(), std::memory_order_relaxed);

    tensor_config_.small_run_target_size = size_t_env_or(
        "ASN_POOL_TENSOR_SMALL_RUN_TARGET_SIZE", nullptr, kDefaultSmallRunTargetSize);
    tensor_config_.large_dirty_decay_epochs = uint64_env_or(
        "ASN_POOL_TENSOR_DECAY_EPOCHS", nullptr, kTensorLargeDirtyDecayEpochs);
    tensor_config_.aggressive_trim = false;
    tensor_config_.hot_bin_min_allocations = size_t_env_or(
        "ASN_POOL_TENSOR_HOT_BIN_MIN_ALLOCS", "ASN_POOL_HOT_BIN_MIN_ALLOCS", 16);
    tensor_config_.hot_bin_max_empty_runs = size_t_env_or(
        "ASN_POOL_TENSOR_HOT_BIN_MAX_EMPTY_RUNS", "ASN_POOL_HOT_BIN_MAX_EMPTY_RUNS", 1);
    tensor_config_.hot_bin_max_idle_generations = size_t_env_or(
        "ASN_POOL_TENSOR_HOT_BIN_MAX_IDLE_GENS", "ASN_POOL_HOT_BIN_MAX_IDLE_GENS", 1);
    tensor_config_.stitched_chunk_cache_max_bytes = size_t_env_or(
        "ASN_POOL_TENSOR_SPOOL_MAX_BYTES", nullptr, kDefaultTensorChunkCacheBytes);

    workspace_config_.small_run_target_size = size_t_env_or(
        "ASN_POOL_WORKSPACE_SMALL_RUN_TARGET_SIZE", nullptr, kWorkspaceSmallRunTargetSize);
    workspace_config_.large_dirty_decay_epochs = uint64_env_or(
        "ASN_POOL_WORKSPACE_DECAY_EPOCHS", nullptr, kWorkspaceLargeDirtyDecayEpochs);
    workspace_config_.aggressive_trim = true;
    workspace_config_.hot_bin_min_allocations = size_t_env_or(
        "ASN_POOL_WORKSPACE_HOT_BIN_MIN_ALLOCS", "ASN_POOL_HOT_BIN_MIN_ALLOCS", 8);
    workspace_config_.hot_bin_max_empty_runs = size_t_env_or(
        "ASN_POOL_WORKSPACE_HOT_BIN_MAX_EMPTY_RUNS", "ASN_POOL_HOT_BIN_MAX_EMPTY_RUNS", 1);
    workspace_config_.hot_bin_max_idle_generations = size_t_env_or(
        "ASN_POOL_WORKSPACE_HOT_BIN_MAX_IDLE_GENS", "ASN_POOL_HOT_BIN_MAX_IDLE_GENS", 0);
    workspace_config_.stitched_chunk_cache_max_bytes = size_t_env_or(
        "ASN_POOL_WORKSPACE_SPOOL_MAX_BYTES", nullptr, kDefaultWorkspaceChunkCacheBytes);
}

size_t MemoryPool::round_up(size_t size) const {
    return ((size + kAlignment - 1) / kAlignment) * kAlignment;
}

size_t MemoryPool::round_up_to_multiple(size_t size, size_t alignment) const {
    if (alignment == 0) {
        return size;
    }
    return ((size + alignment - 1) / alignment) * alignment;
}

size_t MemoryPool::size_class_for(size_t size) const {
    if (size == 0 || size > kSmallAllocationMax) {
        return 0;
    }
    const auto it = std::lower_bound(kSmallSizeClasses.begin(), kSmallSizeClasses.end(), size);
    if (it != kSmallSizeClasses.end()) {
        return *it;
    }
    return 0;
}

void* MemoryPool::malloc(size_t size) {
    return malloc(size, PoolDomain::Tensor);
}

void* MemoryPool::malloc(size_t size, PoolDomain domain_kind) {
    if (size == 0) {
        return nullptr;
    }

    if (!pool_enabled_.load(std::memory_order_relaxed)) {
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("System aclrtMalloc failed");
        }
        return ptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    DomainState& domain = domain_state(domain_kind);
    const DomainConfig& config = domain_config(domain_kind);

    const size_t alloc_size = round_up(size);
    ++domain.stats.allocation_requests;

    const size_t size_class = size_class_for(alloc_size);
    if (size_class != 0) {
        ++domain.stats.small_allocation_requests;
        return allocate_small_locked(domain, config, size_class, size);
    }

    ++domain.stats.large_allocation_requests;
    return allocate_large_locked(domain, config, alloc_size, size);
}

void MemoryPool::free(void* ptr) {
    free(ptr, PoolDomain::Tensor);
}

void MemoryPool::free(void* ptr, PoolDomain domain_kind) {
    if (ptr == nullptr) {
        return;
    }

    if (!pool_enabled_.load(std::memory_order_relaxed)) {
        aclrtFree(ptr);
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    DomainState& primary_domain = domain_state(domain_kind);
    const DomainConfig& primary_config = domain_config(domain_kind);
    if (free_in_domain_locked(primary_domain, primary_config, ptr)) {
        return;
    }

    const PoolDomain fallback_domain =
        domain_kind == PoolDomain::Tensor ? PoolDomain::Workspace : PoolDomain::Tensor;
    DomainState& secondary_domain = domain_state(fallback_domain);
    const DomainConfig& secondary_config = domain_config(fallback_domain);
    (void)free_in_domain_locked(secondary_domain, secondary_config, ptr);
}

void MemoryPool::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_locked(tensor_domain_, tensor_config_, true);
    trim_locked(workspace_domain_, workspace_config_, true);
}

void MemoryPool::clear_cache(PoolDomain domain_kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_locked(domain_state(domain_kind), domain_config(domain_kind), true);
}

void MemoryPool::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_locked(tensor_domain_, tensor_config_, false);
    trim_locked(workspace_domain_, workspace_config_, false);
}

void MemoryPool::trim(PoolDomain domain_kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_locked(domain_state(domain_kind), domain_config(domain_kind), false);
}

void MemoryPool::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    reset_domain_stats_locked(tensor_domain_);
    reset_domain_stats_locked(workspace_domain_);
    peak_active_bytes_total_ = current_total_active_bytes_locked();
}

void MemoryPool::reset_stats(PoolDomain domain_kind) {
    std::lock_guard<std::mutex> lock(mutex_);
    reset_domain_stats_locked(domain_state(domain_kind));
    peak_active_bytes_total_ = current_total_active_bytes_locked();
}

void* MemoryPool::allocate_small_locked(
    DomainState& domain,
    const DomainConfig& config,
    size_t alloc_size,
    size_t requested_size) {
    (void)config;
    auto& bin = domain.small_bins[alloc_size];
    if (bin.size_class == 0) {
        bin.size_class = alloc_size;
    }
    ++bin.alloc_count;

    const bool had_cached_run = bin.partial_run_count > 0 || !bin.empty_runs.empty();
    if (!had_cached_run) {
        allocate_run_locked(domain, config, alloc_size);
    }

    Run* run = select_fullest_partial_run_locked(bin);
    if (run == nullptr) {
        run = bin.empty_runs.back();
    }
    const RunState previous_state = run->state;
    const size_t slot_index = find_free_slot_locked(domain, *run);
    if (slot_index >= run->total_slots) {
        throw std::runtime_error("MemoryPool: no free slot found in non-full run");
    }

    mark_slot_allocated_locked(*run, slot_index);
    --run->free_slots;
    run->first_free_hint = slot_index + 1;
    update_run_state_after_alloc_locked(run, previous_state);

    void* slot_ptr = static_cast<char*>(run->ptr) + (slot_index * run->slot_size);
    if (had_cached_run) {
        ++domain.stats.cache_hits;
        ++domain.stats.small_cache_hits;
    }
    domain.active_requested_sizes[slot_ptr] = requested_size;
    domain.stats.cached_bytes -= run->slot_size;
    domain.stats.active_bytes += run->slot_size;
    domain.stats.active_requested_bytes += requested_size;
    domain.stats.peak_active_bytes = std::max(domain.stats.peak_active_bytes, domain.stats.active_bytes);
    update_total_peak_active_locked();
    return slot_ptr;
}

MemoryPool::Run* MemoryPool::allocate_run_locked(DomainState& domain, const DomainConfig& config, size_t slot_size) {
    size_t target_slots = std::max<size_t>(1, config.small_run_target_size / slot_size);
    size_t slot_count = std::clamp(target_slots, kSmallRunMinSlots, kSmallRunMaxSlots);
    size_t run_size = slot_size * slot_count;

    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, run_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("MemoryPool: small run aclrtMalloc failed ret=" + std::to_string(ret));
    }

    Run* run = new Run();
    run->ptr = ptr;
    run->size = run_size;
    run->slot_size = slot_size;
    run->total_slots = slot_count;
    run->free_slots = slot_count;
    run->first_free_hint = 0;
    run->bitmap.resize((slot_count + 63) / 64, 0);
    run->summary_bitmap.resize((run->bitmap.size() + 63) / 64, 0);
    run->owner = &domain.small_bins[slot_size];
    run->state = RunState::Empty;
    if (run->owner->partial_runs_by_free_slots.size() < slot_count + 1) {
        run->owner->partial_runs_by_free_slots.resize(slot_count + 1);
    }
    for (size_t word_index = 0; word_index < run->bitmap.size(); ++word_index) {
        set_summary_bit_locked(*run, word_index, valid_mask_for_word_locked(*run, word_index) != 0);
    }

    add_run_to_empty_locked(run);
    add_run_to_class_locked(domain, run);
    domain.small_runs_by_base[reinterpret_cast<uintptr_t>(ptr)] = run;
    ++domain.stats.cache_misses;
    ++domain.stats.small_cache_misses;
    ++domain.stats.system_allocations;
    ++domain.stats.small_run_count;
    domain.stats.system_bytes += run_size;
    domain.stats.cached_bytes += run_size;
    return run;
}

MemoryPool::Run* MemoryPool::select_fullest_partial_run_locked(Bin& bin) {
    if (bin.partial_run_count == 0) {
        return nullptr;
    }

    for (size_t free_slots = 1; free_slots < bin.partial_runs_by_free_slots.size(); ++free_slots) {
        auto& bucket = bin.partial_runs_by_free_slots[free_slots];
        if (!bucket.empty()) {
            return bucket.back();
        }
    }

    return nullptr;
}

size_t MemoryPool::stitched_chunk_bytes_locked(DomainState& domain) {
    if (domain.vmm_chunk_bytes != 0) {
        return domain.vmm_chunk_bytes;
    }
    const PoolDomain domain_kind = &domain == &workspace_domain_ ? PoolDomain::Workspace : PoolDomain::Tensor;
    domain.vmm_chunk_bytes = VirtualMemoryManager::instance().chunk_bytes(domain_kind);
    return domain.vmm_chunk_bytes;
}

std::vector<std::shared_ptr<MemoryPool::PhysicalChunk>> MemoryPool::borrow_cached_chunks_locked(
    DomainState& domain,
    size_t chunk_count) {
    std::vector<std::shared_ptr<PhysicalChunk>> chunks;
    chunks.reserve(chunk_count);

    while (chunk_count > 0 && !domain.cached_physical_chunks.empty()) {
        auto chunk = std::move(domain.cached_physical_chunks.back());
        domain.cached_physical_chunks.pop_back();
        chunk->in_chunk_cache = false;
        chunk->last_used_epoch = domain.large_epoch;
        domain.cached_physical_chunk_bytes -= chunk->size;
        domain.stats.cached_bytes -= chunk->size;
        chunks.push_back(std::move(chunk));
        --chunk_count;
    }
    return chunks;
}

void MemoryPool::return_chunks_to_cache_locked(
    DomainState& domain,
    std::vector<std::shared_ptr<PhysicalChunk>>&& chunks) {
    const DomainConfig& config = &domain == &workspace_domain_ ? workspace_config_ : tensor_config_;
    std::vector<std::shared_ptr<PhysicalChunk>> overflow;

    for (auto& chunk : chunks) {
        if (chunk == nullptr) {
            continue;
        }
        chunk->mapped = false;
        chunk->in_chunk_cache = true;
        chunk->last_used_epoch = domain.large_epoch;
        if (domain.cached_physical_chunk_bytes + chunk->size <= config.stitched_chunk_cache_max_bytes) {
            domain.cached_physical_chunk_bytes += chunk->size;
            domain.stats.cached_bytes += chunk->size;
            domain.cached_physical_chunks.push_back(std::move(chunk));
        } else {
            overflow.push_back(std::move(chunk));
        }
    }

    if (!overflow.empty()) {
        free_physical_chunks_locked(domain, std::move(overflow));
    }
}

void MemoryPool::free_physical_chunks_locked(
    DomainState& domain,
    std::vector<std::shared_ptr<PhysicalChunk>>&& chunks) {
    for (auto& chunk : chunks) {
        if (chunk == nullptr || chunk->allocation == nullptr) {
            continue;
        }
        VirtualMemoryManager::instance().free_physical(chunk->allocation);
        ++domain.stats.system_frees;
        domain.stats.system_bytes -= chunk->size;
    }
}

MemoryPool::Block* MemoryPool::map_stitched_segment_locked(
    DomainState& domain,
    const std::vector<std::shared_ptr<PhysicalChunk>>& chunks,
    size_t chunk_bytes) {
    if (chunks.empty()) {
        throw std::runtime_error("map_stitched_segment_locked requires at least one chunk");
    }

    auto reservation_handle = VirtualMemoryManager::instance().reserve(chunks.size() * chunk_bytes);
    size_t mapped_bytes = 0;
    try {
        for (size_t index = 0; index < chunks.size(); ++index) {
            VirtualMemoryManager::instance().map(reservation_handle, index * chunk_bytes, chunks[index]->allocation);
            chunks[index]->mapped = true;
            mapped_bytes += chunk_bytes;
        }
    } catch (...) {
        if (mapped_bytes > 0) {
            (void)VirtualMemoryManager::instance().unmap(reservation_handle, 0, mapped_bytes);
        }
        (void)VirtualMemoryManager::instance().release_reservation(reservation_handle);
        throw;
    }

    auto reservation = std::make_shared<VirtualReservation>();
    reservation->handle = reservation_handle;
    reservation->size = chunks.size() * chunk_bytes;

    auto segment = std::make_shared<StitchedSegment>();
    segment->reservation = reservation;
    segment->chunks = chunks;
    segment->chunk_bytes = chunk_bytes;
    segment->total_bytes = reservation->size;
    segment->mapped_chunk_count = segment->chunks.size();

    Block* block = new Block(reservation_handle->base, segment->total_bytes, false);
    block->is_head = true;
    block->storage_kind = LargeStorageKind::Stitched;
    block->stitched_segment = segment;
    block->chunk_begin = 0;
    block->chunk_count = segment->chunks.size();

    domain.all_blocks[block->ptr] = block;
    domain.live_stitched_segments[block->ptr] = segment;
    ++domain.stats.large_arena_count;
    return block;
}

bool MemoryPool::decompose_free_stitched_blocks_for_request_locked(
    DomainState& domain,
    size_t required_chunk_count) {
    if (domain.cached_physical_chunks.size() >= required_chunk_count) {
        return true;
    }

    std::vector<Block*> releasable_heads;
    for (auto& entry : domain.all_blocks) {
        Block* block = entry.second;
        if (block->storage_kind == LargeStorageKind::Stitched && is_full_free_head_locked(block)) {
            releasable_heads.push_back(block);
        }
    }

    std::sort(releasable_heads.begin(), releasable_heads.end(), [](const Block* lhs, const Block* rhs) {
        return lhs->size > rhs->size;
    });

    for (Block* head : releasable_heads) {
        if (domain.cached_physical_chunks.size() >= required_chunk_count) {
            return true;
        }
        if (!decompose_stitched_head_locked(domain, head)) {
            continue;
        }
    }

    return domain.cached_physical_chunks.size() >= required_chunk_count;
}

void MemoryPool::finalize_large_allocation_locked(DomainState& domain, Block* block, size_t requested_size) {
    block->allocated = true;
    block->last_used_epoch = domain.large_epoch;
    block->decay_epoch = 0;
    block->free_state = LargeBlockState::Active;
    block->free_list_kind = LargeFreeListKind::None;
    domain.active_requested_sizes[block->ptr] = requested_size;
    domain.stats.active_bytes += block->size;
    domain.stats.active_requested_bytes += requested_size;
    domain.stats.peak_active_bytes = std::max(domain.stats.peak_active_bytes, domain.stats.active_bytes);
    update_total_peak_active_locked();
}

void* MemoryPool::allocate_stitched_large_locked(
    DomainState& domain,
    const DomainConfig& config,
    size_t requested_size) {
    (void)config;
    const size_t chunk_bytes = stitched_chunk_bytes_locked(domain);
    const size_t alloc_size = round_up_to_multiple(round_up(requested_size), chunk_bytes);
    const size_t chunk_count = alloc_size / chunk_bytes;

    Block* block = nullptr;
    bool from_cache = false;
    LargeBlockState source_state = LargeBlockState::Active;

    auto dirty_it = domain.dirty_stitched_blocks.lower_bound(alloc_size);
    while (dirty_it != domain.dirty_stitched_blocks.end()) {
        if (!is_quarantined_stitched_block_locked(dirty_it->second)) {
            block = dirty_it->second;
            remove_from_large_free_list(domain, block);
            from_cache = true;
            source_state = LargeBlockState::Dirty;
            ++domain.stats.cache_hits;
            ++domain.stats.large_cache_hits;
            ++domain.stats.stitched_reuse_hits;
            domain.stats.cached_bytes -= block->size;
            break;
        }
        ++dirty_it;
    }
    if (block == nullptr) {
        auto retained_it = domain.retained_stitched_blocks.lower_bound(alloc_size);
        while (retained_it != domain.retained_stitched_blocks.end()) {
            if (!is_quarantined_stitched_block_locked(retained_it->second)) {
                block = retained_it->second;
                remove_from_large_free_list(domain, block);
                from_cache = true;
                source_state = LargeBlockState::Retained;
                ++domain.stats.cache_hits;
                ++domain.stats.large_cache_hits;
                ++domain.stats.stitched_reuse_hits;
                domain.stats.cached_bytes -= block->size;
                break;
            }
            ++retained_it;
        }
    }

    if (block == nullptr) {
        std::vector<std::shared_ptr<PhysicalChunk>> borrowed_chunks = borrow_cached_chunks_locked(domain, chunk_count);
        if (borrowed_chunks.size() < chunk_count) {
            (void)decompose_free_stitched_blocks_for_request_locked(domain, chunk_count);
            auto more_chunks = borrow_cached_chunks_locked(domain, chunk_count - borrowed_chunks.size());
            borrowed_chunks.insert(
                borrowed_chunks.end(),
                std::make_move_iterator(more_chunks.begin()),
                std::make_move_iterator(more_chunks.end()));
        }

        std::vector<std::shared_ptr<PhysicalChunk>> new_chunks;
        new_chunks.reserve(chunk_count > borrowed_chunks.size() ? chunk_count - borrowed_chunks.size() : 0);

        const PoolDomain domain_kind = &domain == &workspace_domain_ ? PoolDomain::Workspace : PoolDomain::Tensor;
        try {
            while (borrowed_chunks.size() + new_chunks.size() < chunk_count) {
                auto allocation = VirtualMemoryManager::instance().allocate_physical(domain_kind, chunk_bytes);
                auto chunk = std::make_shared<PhysicalChunk>();
                chunk->allocation = std::move(allocation);
                chunk->size = chunk_bytes;
                chunk->domain = domain_kind;
                chunk->last_used_epoch = domain.large_epoch;
                new_chunks.push_back(std::move(chunk));
                ++domain.stats.system_allocations;
                domain.stats.system_bytes += chunk_bytes;
            }

            std::vector<std::shared_ptr<PhysicalChunk>> all_chunks;
            all_chunks.reserve(chunk_count);
            all_chunks.insert(
                all_chunks.end(),
                std::make_move_iterator(borrowed_chunks.begin()),
                std::make_move_iterator(borrowed_chunks.end()));
            all_chunks.insert(
                all_chunks.end(),
                std::make_move_iterator(new_chunks.begin()),
                std::make_move_iterator(new_chunks.end()));

            block = map_stitched_segment_locked(domain, all_chunks, chunk_bytes);
        } catch (...) {
            if (!borrowed_chunks.empty()) {
                return_chunks_to_cache_locked(domain, std::move(borrowed_chunks));
            }
            if (!new_chunks.empty()) {
                free_physical_chunks_locked(domain, std::move(new_chunks));
            }
            throw;
        }

        ++domain.stats.cache_misses;
        ++domain.stats.large_cache_misses;
    } else {
        block->last_used_epoch = domain.large_epoch;
    }

    if (block->size >= alloc_size + chunk_bytes) {
        const size_t remaining_size = block->size - alloc_size;
        void* remaining_ptr = static_cast<char*>(block->ptr) + alloc_size;

        Block* remaining = new Block(remaining_ptr, remaining_size, false);
        remaining->last_used_epoch = domain.large_epoch;
        remaining->prev = block;
        remaining->next = block->next;
        if (remaining->next != nullptr) {
            remaining->next->prev = remaining;
        }
        remaining->storage_kind = LargeStorageKind::Stitched;
        remaining->stitched_segment = block->stitched_segment;
        remaining->chunk_begin = block->chunk_begin + chunk_count;
        remaining->chunk_count = block->chunk_count - chunk_count;
        block->next = remaining;
        block->size = alloc_size;
        block->chunk_count = chunk_count;

        domain.all_blocks[remaining_ptr] = remaining;
        add_to_large_free_list(domain, config, remaining, from_cache ? source_state : LargeBlockState::Retained);
        domain.stats.cached_bytes += remaining_size;
        ++domain.stats.large_split_count;
    }

    finalize_large_allocation_locked(domain, block, requested_size);
    return block->ptr;
}

void* MemoryPool::allocate_legacy_large_locked(
    DomainState& domain,
    const DomainConfig& config,
    size_t alloc_size,
    size_t requested_size) {
    Block* block = nullptr;
    bool from_cache = false;
    LargeBlockState source_state = LargeBlockState::Active;

    auto dirty_it = domain.dirty_large_blocks.lower_bound(alloc_size);
    if (dirty_it != domain.dirty_large_blocks.end()) {
        block = dirty_it->second;
        remove_from_large_free_list(domain, block);
        from_cache = true;
        source_state = LargeBlockState::Dirty;
        ++domain.stats.cache_hits;
        ++domain.stats.large_cache_hits;
        domain.stats.cached_bytes -= block->size;
    } else {
        auto retained_it = domain.retained_large_blocks.lower_bound(alloc_size);
        if (retained_it != domain.retained_large_blocks.end()) {
            block = retained_it->second;
            remove_from_large_free_list(domain, block);
            from_cache = true;
            source_state = LargeBlockState::Retained;
            ++domain.stats.cache_hits;
            ++domain.stats.large_cache_hits;
            domain.stats.cached_bytes -= block->size;
        }
    }

    if (block == nullptr) {
        ++domain.stats.cache_misses;
        ++domain.stats.large_cache_misses;

        const size_t system_alloc_size = alloc_size > kDefaultArenaSize ? alloc_size : kDefaultArenaSize;
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, system_alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("MemoryPool: aclrtMalloc failed ret=" + std::to_string(ret));
        }

        block = new Block(ptr, system_alloc_size, false);
        block->is_head = true;
        block->storage_kind = LargeStorageKind::LegacyContiguous;
        block->last_used_epoch = domain.large_epoch;
        domain.all_blocks[ptr] = block;

        ++domain.stats.system_allocations;
        ++domain.stats.large_arena_count;
        domain.stats.system_bytes += system_alloc_size;
    } else {
        block->last_used_epoch = domain.large_epoch;
    }

    if (block->size >= alloc_size + kLargeMinSplitSize) {
        const size_t remaining_size = block->size - alloc_size;
        void* remaining_ptr = static_cast<char*>(block->ptr) + alloc_size;

        Block* remaining = new Block(remaining_ptr, remaining_size, false);
        remaining->last_used_epoch = domain.large_epoch;
        remaining->prev = block;
        remaining->next = block->next;
        if (remaining->next != nullptr) {
            remaining->next->prev = remaining;
        }
        remaining->storage_kind = LargeStorageKind::LegacyContiguous;
        block->next = remaining;
        block->size = alloc_size;

        domain.all_blocks[remaining_ptr] = remaining;
        add_to_large_free_list(domain, config, remaining, from_cache ? source_state : LargeBlockState::Retained);
        domain.stats.cached_bytes += remaining_size;
        ++domain.stats.large_split_count;
    }

    finalize_large_allocation_locked(domain, block, requested_size);
    return block->ptr;
}

void* MemoryPool::allocate_large_locked(
    DomainState& domain,
    const DomainConfig& config,
    size_t alloc_size,
    size_t requested_size) {
    ++domain.large_epoch;
    decay_large_blocks_locked(domain, config, false);

    const PoolDomain domain_kind = &domain == &workspace_domain_ ? PoolDomain::Workspace : PoolDomain::Tensor;
    auto& vmm = VirtualMemoryManager::instance();
    const bool vmm_enabled = vmm.enabled(domain_kind);
    if (vmm_enabled) {
        try {
            return allocate_stitched_large_locked(domain, config, requested_size);
        } catch (const std::exception& e) {
            if (vmm.should_report_fallback_once()) {
                std::fprintf(stderr, "[MemoryPool] VMM stitched allocation failed: %s\n", e.what());
            }
            ++domain.stats.vmm_fallback_allocations;
        } catch (...) {
            if (vmm.should_report_fallback_once()) {
                std::fprintf(stderr, "[MemoryPool] VMM stitched allocation failed with unknown exception\n");
            }
            ++domain.stats.vmm_fallback_allocations;
        }
    } else {
        if (vmm.should_report_fallback_once()) {
            const std::string error = vmm.last_error();
            if (!error.empty()) {
                std::fprintf(stderr, "[MemoryPool] VMM disabled, falling back to legacy allocator: %s\n", error.c_str());
            } else {
                std::fprintf(stderr, "[MemoryPool] VMM disabled, falling back to legacy allocator\n");
            }
        }
        ++domain.stats.vmm_fallback_allocations;
    }

    return allocate_legacy_large_locked(domain, config, alloc_size, requested_size);
}

bool MemoryPool::free_in_domain_locked(DomainState& domain, const DomainConfig& config, void* ptr) {
    if (Run* run = find_small_run_locked(domain, ptr)) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(run->ptr);
        const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        const size_t offset = static_cast<size_t>(addr - base);
        if (offset % run->slot_size != 0) {
            return true;
        }

        const size_t slot_index = offset / run->slot_size;
        if (slot_index >= run->total_slots) {
            return true;
        }

        ++domain.stats.free_requests;
        auto requested_it = domain.active_requested_sizes.find(ptr);
        const size_t requested_size =
            requested_it != domain.active_requested_sizes.end() ? requested_it->second : run->slot_size;
        free_small_locked(domain, run, slot_index, ptr, requested_size);
        return true;
    }

    auto it = domain.all_blocks.find(ptr);
    if (it == domain.all_blocks.end()) {
        return false;
    }

    ++domain.stats.free_requests;
    Block* block = it->second;
    if (!block->allocated) {
        return true;
    }
    auto requested_it = domain.active_requested_sizes.find(ptr);
    const size_t requested_size =
        requested_it != domain.active_requested_sizes.end() ? requested_it->second : block->size;
    free_large_locked(domain, config, block, requested_size);
    return true;
}

void MemoryPool::free_small_locked(DomainState& domain, Run* run, size_t slot_index, void* ptr, size_t requested_size) {
    if (!is_slot_allocated_locked(*run, slot_index)) {
        return;
    }

    const RunState previous_state = run->state;
    mark_slot_free_locked(*run, slot_index);
    ++run->free_slots;
    run->first_free_hint = std::min(run->first_free_hint, slot_index);
    update_run_state_after_free_locked(run, previous_state);

    domain.active_requested_sizes.erase(ptr);
    domain.stats.active_bytes -= run->slot_size;
    domain.stats.active_requested_bytes -= requested_size;
    domain.stats.cached_bytes += run->slot_size;
}

bool MemoryPool::can_merge_blocks_locked(const Block* left, const Block* right) const {
    if (left == nullptr || right == nullptr) {
        return false;
    }
    if (left->storage_kind != right->storage_kind) {
        return false;
    }
    if (static_cast<const char*>(right->ptr) != static_cast<const char*>(left->ptr) + left->size) {
        return false;
    }

    if (left->storage_kind == LargeStorageKind::Stitched) {
        return left->stitched_segment == right->stitched_segment &&
               left->chunk_begin + left->chunk_count == right->chunk_begin;
    }

    return true;
}

bool MemoryPool::is_full_free_head_locked(const Block* block) const {
    return block != nullptr &&
           !block->allocated &&
           block->is_head &&
           block->prev == nullptr &&
           block->next == nullptr &&
           !is_quarantined_stitched_block_locked(block);
}

bool MemoryPool::is_quarantined_stitched_block_locked(const Block* block) const {
    return block != nullptr &&
           block->storage_kind == LargeStorageKind::Stitched &&
           block->stitched_segment != nullptr &&
           block->stitched_segment->state == StitchedSegment::State::Quarantined;
}

void MemoryPool::free_large_locked(DomainState& domain, const DomainConfig& config, Block* block, size_t requested_size) {
    const size_t released_size = block->size;
    void* original_ptr = block->ptr;
    ++domain.large_epoch;
    block->allocated = false;
    block->last_used_epoch = domain.large_epoch;
    domain.active_requested_sizes.erase(original_ptr);

    Block* merged = try_merge_locked(domain, block);
    merged->last_used_epoch = domain.large_epoch;
    add_to_large_free_list(domain, config, merged, LargeBlockState::Dirty);

    domain.stats.active_bytes -= released_size;
    domain.stats.active_requested_bytes -= requested_size;
    domain.stats.cached_bytes += released_size;
    decay_large_blocks_locked(domain, config, false);
}

MemoryPool::Block* MemoryPool::try_merge_locked(DomainState& domain, Block* block) {
    if (block->next != nullptr && !block->next->allocated && can_merge_blocks_locked(block, block->next)) {
        Block* next_block = block->next;
        remove_from_large_free_list(domain, next_block);

        block->size += next_block->size;
        block->next = next_block->next;
        if (block->next != nullptr) {
            block->next->prev = block;
        }
        if (block->storage_kind == LargeStorageKind::Stitched) {
            block->chunk_count += next_block->chunk_count;
        }

        domain.all_blocks.erase(next_block->ptr);
        delete next_block;
        ++domain.stats.large_merge_count;
    }

    if (block->prev != nullptr && !block->prev->allocated && can_merge_blocks_locked(block->prev, block)) {
        Block* prev_block = block->prev;
        remove_from_large_free_list(domain, prev_block);

        prev_block->size += block->size;
        prev_block->next = block->next;
        if (prev_block->next != nullptr) {
            prev_block->next->prev = prev_block;
        }
        if (prev_block->storage_kind == LargeStorageKind::Stitched) {
            prev_block->chunk_count += block->chunk_count;
        }

        domain.all_blocks.erase(block->ptr);
        delete block;
        ++domain.stats.large_merge_count;
        return prev_block;
    }

    return block;
}

void MemoryPool::add_to_large_free_list(
    DomainState& domain,
    const DomainConfig& config,
    Block* block,
    LargeBlockState state) {
    (void)config;
    block->free_state = state;
    if (state == LargeBlockState::Dirty) {
        block->decay_epoch = block->last_used_epoch + config.large_dirty_decay_epochs;
    } else {
        block->decay_epoch = 0;
    }

    LargeFreeList* free_blocks = nullptr;
    if (block->storage_kind == LargeStorageKind::Stitched) {
        if (state == LargeBlockState::Dirty) {
            free_blocks = &domain.dirty_stitched_blocks;
            block->free_list_kind = LargeFreeListKind::StitchedDirty;
            domain.dirty_stitched_decay_buckets[block->decay_epoch].push_back(block->ptr);
        } else {
            free_blocks = &domain.retained_stitched_blocks;
            block->free_list_kind = LargeFreeListKind::StitchedRetained;
        }
    } else {
        if (state == LargeBlockState::Dirty) {
            free_blocks = &domain.dirty_large_blocks;
            block->free_list_kind = LargeFreeListKind::LegacyDirty;
            domain.dirty_large_decay_buckets[block->decay_epoch].push_back(block->ptr);
        } else {
            free_blocks = &domain.retained_large_blocks;
            block->free_list_kind = LargeFreeListKind::LegacyRetained;
        }
    }

    block->free_list_iter = free_blocks->insert({block->size, block});
    block->in_free_list = true;
    update_large_block_state_counters(domain, state, block->size, true);
}

void MemoryPool::remove_from_large_free_list(DomainState& domain, Block* block) {
    if (block == nullptr || !block->in_free_list) {
        return;
    }

    LargeFreeList* free_blocks = nullptr;
    switch (block->free_list_kind) {
    case LargeFreeListKind::LegacyDirty:
        free_blocks = &domain.dirty_large_blocks;
        break;
    case LargeFreeListKind::LegacyRetained:
        free_blocks = &domain.retained_large_blocks;
        break;
    case LargeFreeListKind::StitchedDirty:
        free_blocks = &domain.dirty_stitched_blocks;
        break;
    case LargeFreeListKind::StitchedRetained:
        free_blocks = &domain.retained_stitched_blocks;
        break;
    case LargeFreeListKind::None:
        break;
    }

    if (free_blocks != nullptr) {
        free_blocks->erase(block->free_list_iter);
        update_large_block_state_counters(domain, block->free_state, block->size, false);
    }

    block->free_list_iter = LargeFreeList::iterator{};
    block->free_list_kind = LargeFreeListKind::None;
    block->free_state = LargeBlockState::Active;
    block->in_free_list = false;
    block->decay_epoch = 0;
}

void MemoryPool::release_run_locked(DomainState& domain, Run* run) {
    remove_run_from_partial_locked(run);
    remove_run_from_empty_locked(run);
    remove_run_from_class_locked(domain, run);
    domain.small_runs_by_base.erase(reinterpret_cast<uintptr_t>(run->ptr));

    aclrtFree(run->ptr);
    ++domain.stats.system_frees;
    --domain.stats.small_run_count;
    domain.stats.system_bytes -= run->size;
    domain.stats.cached_bytes -= run->size;
    delete run;
}

bool MemoryPool::decompose_stitched_head_locked(DomainState& domain, Block* block) {
    if (block == nullptr || block->storage_kind != LargeStorageKind::Stitched || !is_full_free_head_locked(block)) {
        return false;
    }

    auto segment = block->stitched_segment;
    if (segment == nullptr || segment->reservation == nullptr || segment->reservation->handle == nullptr) {
        return false;
    }

    remove_from_large_free_list(domain, block);
    domain.stats.cached_bytes -= block->size;

    const auto unmap_result = VirtualMemoryManager::instance().unmap(segment->reservation->handle, 0, segment->total_bytes);
    if (unmap_result.status != ACL_SUCCESS) {
        if (unmap_result.unmapped_chunk_count == 0) {
            const DomainConfig& config = &domain == &workspace_domain_ ? workspace_config_ : tensor_config_;
            add_to_large_free_list(domain, config, block, LargeBlockState::Retained);
            domain.stats.cached_bytes += block->size;
            return false;
        }
        segment->state = StitchedSegment::State::Quarantined;
        segment->mapped_chunk_count = segment->chunks.size() - unmap_result.unmapped_chunk_count;
        return false;
    }

    for (size_t index = 0; index < unmap_result.unmapped_chunk_count && index < segment->chunks.size(); ++index) {
        if (segment->chunks[index] != nullptr) {
            segment->chunks[index]->mapped = false;
        }
    }
    segment->mapped_chunk_count = segment->chunks.size() >= unmap_result.unmapped_chunk_count
        ? (segment->chunks.size() - unmap_result.unmapped_chunk_count)
        : 0;

    const aclError release_ret = VirtualMemoryManager::instance().release_reservation(segment->reservation->handle);
    if (release_ret != ACL_SUCCESS) {
        segment->state = StitchedSegment::State::Quarantined;
        segment->mapped_chunk_count = 0;
        return false;
    }

    for (auto& chunk : segment->chunks) {
        if (chunk != nullptr) {
            chunk->mapped = false;
        }
    }
    segment->mapped_chunk_count = 0;

    std::vector<std::shared_ptr<PhysicalChunk>> chunks = std::move(segment->chunks);
    domain.live_stitched_segments.erase(block->ptr);
    domain.all_blocks.erase(block->ptr);
    --domain.stats.large_arena_count;
    delete block;

    return_chunks_to_cache_locked(domain, std::move(chunks));
    return true;
}

void MemoryPool::release_large_head_locked(DomainState& domain, Block* block) {
    if (block == nullptr) {
        return;
    }

    if (block->storage_kind == LargeStorageKind::Stitched) {
        if (!decompose_stitched_head_locked(domain, block)) {
            return;
        }
        return;
    }

    remove_from_large_free_list(domain, block);
    aclrtFree(block->ptr);
    domain.all_blocks.erase(block->ptr);

    ++domain.stats.system_frees;
    --domain.stats.large_arena_count;
    domain.stats.system_bytes -= block->size;
    domain.stats.cached_bytes -= block->size;
    delete block;
}

void MemoryPool::release_cached_physical_chunks_locked(DomainState& domain) {
    std::vector<std::shared_ptr<PhysicalChunk>> chunks;
    chunks.swap(domain.cached_physical_chunks);
    if (chunks.empty()) {
        return;
    }

    domain.stats.cached_bytes -= domain.cached_physical_chunk_bytes;
    domain.cached_physical_chunk_bytes = 0;
    free_physical_chunks_locked(domain, std::move(chunks));
}

void MemoryPool::decay_large_blocks_locked(DomainState& domain, const DomainConfig& config, bool force_all) {
    auto decay_one = [&](LargeFreeList& dirty_blocks,
                         LargeFreeList& retained_blocks,
                         std::map<uint64_t, std::vector<void*>>& decay_buckets,
                         LargeStorageKind storage_kind) {
        std::vector<Block*> to_retain;
        if (force_all) {
            to_retain.reserve(dirty_blocks.size());
            for (const auto& entry : dirty_blocks) {
                if (!is_quarantined_stitched_block_locked(entry.second)) {
                    to_retain.push_back(entry.second);
                }
            }
            decay_buckets.clear();
        } else {
            auto bucket_it = decay_buckets.begin();
            while (bucket_it != decay_buckets.end() && bucket_it->first <= domain.large_epoch) {
                const uint64_t decay_epoch = bucket_it->first;
                auto pending_blocks = std::move(bucket_it->second);
                bucket_it = decay_buckets.erase(bucket_it);

                for (void* block_ptr : pending_blocks) {
                    auto block_it = domain.all_blocks.find(block_ptr);
                    if (block_it == domain.all_blocks.end()) {
                        continue;
                    }

                    Block* block = block_it->second;
                    if (block->storage_kind != storage_kind) {
                        continue;
                    }
                    if (is_quarantined_stitched_block_locked(block)) {
                        continue;
                    }
                    if (large_block_state(*block) != LargeBlockState::Dirty) {
                        continue;
                    }
                    if (block->decay_epoch != decay_epoch) {
                        continue;
                    }

                    to_retain.push_back(block);
                }
            }
        }

        for (Block* block : to_retain) {
            remove_from_large_free_list(domain, block);
            block->free_list_kind = LargeFreeListKind::None;
            block->free_state = LargeBlockState::Active;
            block->in_free_list = false;
            block->decay_epoch = 0;
            if (storage_kind == LargeStorageKind::Stitched) {
                block->free_list_iter = retained_blocks.insert({block->size, block});
                block->free_list_kind = LargeFreeListKind::StitchedRetained;
            } else {
                block->free_list_iter = retained_blocks.insert({block->size, block});
                block->free_list_kind = LargeFreeListKind::LegacyRetained;
            }
            block->free_state = LargeBlockState::Retained;
            block->in_free_list = true;
            update_large_block_state_counters(domain, LargeBlockState::Retained, block->size, true);
            ++domain.stats.large_decay_count;
        }
    };

    decay_one(domain.dirty_large_blocks, domain.retained_large_blocks, domain.dirty_large_decay_buckets, LargeStorageKind::LegacyContiguous);
    decay_one(domain.dirty_stitched_blocks, domain.retained_stitched_blocks, domain.dirty_stitched_decay_buckets, LargeStorageKind::Stitched);
}

void MemoryPool::trim_locked(DomainState& domain, const DomainConfig& config, bool release_all_large) {
    decay_large_blocks_locked(domain, config, true);

    std::vector<Run*> releasable_runs;
    for (auto& entry : domain.small_bins) {
        auto& bin = entry.second;
        const size_t recent_allocations = bin.alloc_count - bin.last_trim_alloc_count;
        const bool hot_this_trim = recent_allocations >= config.hot_bin_min_allocations;
        if (hot_this_trim) {
            bin.idle_generations = 0;
        } else {
            ++bin.idle_generations;
        }
        bin.last_trim_alloc_count = bin.alloc_count;

        size_t retain_count = 0;
        if (!release_all_large &&
            bin.idle_generations <= config.hot_bin_max_idle_generations) {
            retain_count = std::min(config.hot_bin_max_empty_runs, bin.empty_runs.size());
        }

        const size_t releasable_count = bin.empty_runs.size() > retain_count
            ? (bin.empty_runs.size() - retain_count)
            : 0;

        for (size_t i = 0; i < bin.empty_runs.size(); ++i) {
            Run* run = bin.empty_runs[i];
            run->hot_retained = i >= releasable_count;
            if (i < releasable_count) {
                releasable_runs.push_back(run);
            }
        }
    }

    for (Run* run : releasable_runs) {
        release_run_locked(domain, run);
    }

    std::vector<Block*> releasable_heads;
    for (auto& entry : domain.all_blocks) {
        Block* block = entry.second;
        if (!is_full_free_head_locked(block)) {
            continue;
        }

        const LargeBlockState state = large_block_state(*block);
        if (release_all_large || config.aggressive_trim || state == LargeBlockState::Retained) {
            releasable_heads.push_back(block);
        }
    }

    for (Block* head : releasable_heads) {
        domain.stats.trimmed_bytes += head->size;
        release_large_head_locked(domain, head);
    }

    if (release_all_large) {
        release_cached_physical_chunks_locked(domain);
    }
}

MemoryPoolStats MemoryPool::snapshot_domain_stats_locked(const DomainState& domain) const {
    MemoryPoolStats snapshot = domain.stats;
    snapshot.empty_small_run_count = 0;
    snapshot.partial_small_run_count = 0;
    snapshot.full_small_run_count = 0;
    snapshot.small_slots_total = 0;
    snapshot.small_slots_free = 0;
    snapshot.small_free_bytes = 0;
    snapshot.largest_free_run_bytes = 0;
    snapshot.largest_free_block_bytes = 0;
    snapshot.dirty_large_block_count = domain.dirty_large_blocks.size() + domain.dirty_stitched_blocks.size();
    snapshot.retained_large_block_count = domain.retained_large_blocks.size() + domain.retained_stitched_blocks.size();
    snapshot.hot_retained_empty_runs = 0;
    snapshot.chunk_cache_bytes = domain.cached_physical_chunk_bytes;
    snapshot.chunk_cache_count = domain.cached_physical_chunks.size();
    snapshot.stitched_segment_count = domain.live_stitched_segments.size();
    snapshot.vmm_internal_fragmentation_bytes = 0;

    for (const auto& entry : domain.runs_by_class) {
        for (Run* run : entry.second) {
            snapshot.small_slots_total += run->total_slots;
            snapshot.small_slots_free += run->free_slots;
            const size_t free_bytes = run->free_slots * run->slot_size;
            snapshot.small_free_bytes += free_bytes;
            snapshot.largest_free_run_bytes = std::max(snapshot.largest_free_run_bytes, free_bytes);
            switch (run->state) {
            case RunState::Empty:
                ++snapshot.empty_small_run_count;
                if (run->hot_retained) {
                    ++snapshot.hot_retained_empty_runs;
                }
                break;
            case RunState::Partial:
                ++snapshot.partial_small_run_count;
                break;
            case RunState::Full:
                ++snapshot.full_small_run_count;
                break;
            }
        }
    }

    auto update_largest = [&snapshot](const LargeFreeList& free_list) {
        if (!free_list.empty()) {
            snapshot.largest_free_block_bytes = std::max(snapshot.largest_free_block_bytes, free_list.rbegin()->first);
        }
    };
    update_largest(domain.dirty_large_blocks);
    update_largest(domain.retained_large_blocks);
    update_largest(domain.dirty_stitched_blocks);
    update_largest(domain.retained_stitched_blocks);

    for (const auto& entry : domain.active_requested_sizes) {
        auto block_it = domain.all_blocks.find(entry.first);
        if (block_it == domain.all_blocks.end()) {
            continue;
        }
        const Block* block = block_it->second;
        if (block->allocated && block->storage_kind == LargeStorageKind::Stitched) {
            snapshot.vmm_internal_fragmentation_bytes += block->size >= entry.second ? block->size - entry.second : 0;
        }
    }

    snapshot.internal_fragmentation_bytes =
        snapshot.active_bytes >= snapshot.active_requested_bytes
            ? snapshot.active_bytes - snapshot.active_requested_bytes
            : 0;
    snapshot.small_external_fragmentation_bytes =
        snapshot.small_free_bytes >= snapshot.largest_free_run_bytes
            ? snapshot.small_free_bytes - snapshot.largest_free_run_bytes
            : 0;
    const size_t large_free_bytes = snapshot.dirty_large_bytes + snapshot.retained_large_bytes;
    snapshot.large_external_fragmentation_bytes =
        large_free_bytes >= snapshot.largest_free_block_bytes
            ? large_free_bytes - snapshot.largest_free_block_bytes
            : 0;
    snapshot.external_fragmentation_bytes =
        snapshot.small_external_fragmentation_bytes + snapshot.large_external_fragmentation_bytes;
    snapshot.internal_fragmentation_ratio_pct = snapshot.active_bytes > 0
        ? (static_cast<double>(snapshot.internal_fragmentation_bytes) * 100.0) / static_cast<double>(snapshot.active_bytes)
        : 0.0;
    snapshot.small_external_fragmentation_ratio_pct = snapshot.small_free_bytes > 0
        ? (static_cast<double>(snapshot.small_external_fragmentation_bytes) * 100.0) / static_cast<double>(snapshot.small_free_bytes)
        : 0.0;
    snapshot.large_external_fragmentation_ratio_pct = large_free_bytes > 0
        ? (static_cast<double>(snapshot.large_external_fragmentation_bytes) * 100.0) / static_cast<double>(large_free_bytes)
        : 0.0;
    const size_t total_free_bytes = snapshot.small_free_bytes + large_free_bytes;
    snapshot.external_fragmentation_ratio_pct = total_free_bytes > 0
        ? (static_cast<double>(snapshot.external_fragmentation_bytes) * 100.0) / static_cast<double>(total_free_bytes)
        : 0.0;
    const size_t active_slots = snapshot.small_slots_total >= snapshot.small_slots_free
        ? snapshot.small_slots_total - snapshot.small_slots_free
        : 0;
    snapshot.run_utilization_pct = snapshot.small_slots_total > 0
        ? (static_cast<double>(active_slots) * 100.0) / static_cast<double>(snapshot.small_slots_total)
        : 0.0;

    return snapshot;
}

void MemoryPool::reset_domain_stats_locked(DomainState& domain) {
    MemoryPoolStats baseline;
    baseline.active_bytes = domain.stats.active_bytes;
    baseline.active_requested_bytes = domain.stats.active_requested_bytes;
    baseline.cached_bytes = domain.stats.cached_bytes;
    baseline.system_bytes = domain.stats.system_bytes;
    baseline.peak_active_bytes = domain.stats.active_bytes;
    baseline.small_run_count = domain.stats.small_run_count;
    baseline.large_arena_count = domain.stats.large_arena_count;
    baseline.dirty_large_bytes = domain.stats.dirty_large_bytes;
    baseline.retained_large_bytes = domain.stats.retained_large_bytes;
    domain.stats = baseline;
}

void MemoryPool::accumulate_stats(MemoryPoolStats& dst, const MemoryPoolStats& src) {
    dst.allocation_requests += src.allocation_requests;
    dst.free_requests += src.free_requests;
    dst.cache_hits += src.cache_hits;
    dst.cache_misses += src.cache_misses;
    dst.small_allocation_requests += src.small_allocation_requests;
    dst.small_cache_hits += src.small_cache_hits;
    dst.small_cache_misses += src.small_cache_misses;
    dst.large_allocation_requests += src.large_allocation_requests;
    dst.large_cache_hits += src.large_cache_hits;
    dst.large_cache_misses += src.large_cache_misses;
    dst.system_allocations += src.system_allocations;
    dst.system_frees += src.system_frees;
    dst.active_bytes += src.active_bytes;
    dst.active_requested_bytes += src.active_requested_bytes;
    dst.cached_bytes += src.cached_bytes;
    dst.system_bytes += src.system_bytes;
    dst.peak_active_bytes = std::max(dst.peak_active_bytes, src.peak_active_bytes);
    dst.small_run_count += src.small_run_count;
    dst.large_arena_count += src.large_arena_count;
    dst.empty_small_run_count += src.empty_small_run_count;
    dst.partial_small_run_count += src.partial_small_run_count;
    dst.full_small_run_count += src.full_small_run_count;
    dst.small_slots_total += src.small_slots_total;
    dst.small_slots_free += src.small_slots_free;
    dst.bitmap_words_scanned += src.bitmap_words_scanned;
    dst.summary_words_scanned += src.summary_words_scanned;
    dst.dirty_large_block_count += src.dirty_large_block_count;
    dst.retained_large_block_count += src.retained_large_block_count;
    dst.dirty_large_bytes += src.dirty_large_bytes;
    dst.retained_large_bytes += src.retained_large_bytes;
    dst.large_split_count += src.large_split_count;
    dst.large_merge_count += src.large_merge_count;
    dst.large_decay_count += src.large_decay_count;
    dst.trimmed_bytes += src.trimmed_bytes;
    dst.small_free_bytes += src.small_free_bytes;
    dst.largest_free_run_bytes = std::max(dst.largest_free_run_bytes, src.largest_free_run_bytes);
    dst.largest_free_block_bytes = std::max(dst.largest_free_block_bytes, src.largest_free_block_bytes);
    dst.internal_fragmentation_bytes += src.internal_fragmentation_bytes;
    dst.vmm_internal_fragmentation_bytes += src.vmm_internal_fragmentation_bytes;
    dst.small_external_fragmentation_bytes += src.small_external_fragmentation_bytes;
    dst.large_external_fragmentation_bytes += src.large_external_fragmentation_bytes;
    dst.external_fragmentation_bytes += src.external_fragmentation_bytes;
    dst.hot_retained_empty_runs += src.hot_retained_empty_runs;
    dst.stitched_reuse_hits += src.stitched_reuse_hits;
    dst.vmm_fallback_allocations += src.vmm_fallback_allocations;
    dst.chunk_cache_bytes += src.chunk_cache_bytes;
    dst.chunk_cache_count += src.chunk_cache_count;
    dst.stitched_segment_count += src.stitched_segment_count;
    dst.internal_fragmentation_ratio_pct = dst.active_bytes > 0
        ? (static_cast<double>(dst.internal_fragmentation_bytes) * 100.0) / static_cast<double>(dst.active_bytes)
        : 0.0;
    dst.small_external_fragmentation_ratio_pct = dst.small_free_bytes > 0
        ? (static_cast<double>(dst.small_external_fragmentation_bytes) * 100.0) / static_cast<double>(dst.small_free_bytes)
        : 0.0;
    const size_t total_large_free_bytes = dst.dirty_large_bytes + dst.retained_large_bytes;
    dst.large_external_fragmentation_ratio_pct = total_large_free_bytes > 0
        ? (static_cast<double>(dst.large_external_fragmentation_bytes) * 100.0) / static_cast<double>(total_large_free_bytes)
        : 0.0;
    const size_t total_free_bytes = dst.small_free_bytes + total_large_free_bytes;
    dst.external_fragmentation_ratio_pct = total_free_bytes > 0
        ? (static_cast<double>(dst.external_fragmentation_bytes) * 100.0) / static_cast<double>(total_free_bytes)
        : 0.0;
    const size_t total_slots = dst.small_slots_total;
    const size_t free_slots = dst.small_slots_free;
    const size_t active_slots = total_slots >= free_slots ? total_slots - free_slots : 0;
    dst.run_utilization_pct = total_slots > 0
        ? (static_cast<double>(active_slots) * 100.0) / static_cast<double>(total_slots)
        : 0.0;
}

MemoryPoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryPoolStats aggregate;
    accumulate_stats(aggregate, snapshot_domain_stats_locked(tensor_domain_));
    accumulate_stats(aggregate, snapshot_domain_stats_locked(workspace_domain_));
    aggregate.peak_active_bytes = peak_active_bytes_total_;
    return aggregate;
}

MemoryPoolStats MemoryPool::stats(PoolDomain domain_kind) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return snapshot_domain_stats_locked(domain_state(domain_kind));
}

MemoryPool::Run* MemoryPool::find_small_run_locked(const DomainState& domain, void* ptr) const {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    auto it = domain.small_runs_by_base.upper_bound(addr);
    if (it == domain.small_runs_by_base.begin()) {
        return nullptr;
    }

    --it;
    Run* run = it->second;
    const uintptr_t base = it->first;
    const uintptr_t end = base + run->size;
    return addr < end ? run : nullptr;
}

size_t MemoryPool::find_free_slot_locked(DomainState& domain, const Run& run) {
    if (run.total_slots == 0) {
        return 0;
    }

    const size_t word_count = run.bitmap.size();
    const size_t start_slot = std::min(run.first_free_hint, run.total_slots);
    const size_t start_word = start_slot / 64;
    const size_t start_bit = start_slot % 64;

    auto scan_word = [this, &domain, &run](size_t word_index, uint64_t free_bits) -> size_t {
        ++domain.stats.bitmap_words_scanned;
        if (free_bits == 0) {
            return run.total_slots;
        }

        const size_t bit_index = first_free_bit_index_locked(free_bits);
        const size_t slot_index = (word_index * 64) + bit_index;
        return slot_index < run.total_slots ? slot_index : run.total_slots;
    };

    if (start_word < word_count) {
        uint64_t free_bits = (~run.bitmap[start_word]) & valid_mask_for_word_locked(run, start_word);
        if (start_bit != 0) {
            const uint64_t prefix_mask = (uint64_t{1} << start_bit) - 1;
            free_bits &= ~prefix_mask;
        }
        const size_t slot_index = scan_word(start_word, free_bits);
        if (slot_index < run.total_slots) {
            return slot_index;
        }
    }

    const size_t forward_word = find_free_word_locked(domain, run, std::min(start_word + 1, word_count));
    if (forward_word < word_count) {
        const uint64_t free_bits = (~run.bitmap[forward_word]) & valid_mask_for_word_locked(run, forward_word);
        const size_t slot_index = scan_word(forward_word, free_bits);
        if (slot_index < run.total_slots) {
            return slot_index;
        }
    }

    const size_t wrapped_word = find_free_word_locked(domain, run, 0);
    if (wrapped_word < std::min(start_word, word_count)) {
        const uint64_t free_bits = (~run.bitmap[wrapped_word]) & valid_mask_for_word_locked(run, wrapped_word);
        const size_t slot_index = scan_word(wrapped_word, free_bits);
        if (slot_index < run.total_slots) {
            return slot_index;
        }
    }

    if (start_word < word_count && start_bit != 0) {
        uint64_t free_bits = (~run.bitmap[start_word]) & valid_mask_for_word_locked(run, start_word);
        free_bits &= ((uint64_t{1} << start_bit) - 1);
        const size_t slot_index = scan_word(start_word, free_bits);
        if (slot_index < run.total_slots) {
            return slot_index;
        }
    }

    return run.total_slots;
}

size_t MemoryPool::find_free_word_locked(DomainState& domain, const Run& run, size_t start_word) {
    const size_t word_count = run.bitmap.size();
    if (start_word >= word_count) {
        return word_count;
    }

    const size_t summary_word_count = run.summary_bitmap.size();
    const size_t start_summary_word = start_word / 64;
    const size_t start_summary_bit = start_word % 64;

    for (size_t summary_word_index = start_summary_word; summary_word_index < summary_word_count; ++summary_word_index) {
        ++domain.stats.summary_words_scanned;
        uint64_t candidate_words = run.summary_bitmap[summary_word_index];
        if (summary_word_index == start_summary_word && start_summary_bit != 0) {
            const uint64_t prefix_mask = (uint64_t{1} << start_summary_bit) - 1;
            candidate_words &= ~prefix_mask;
        }
        if (candidate_words == 0) {
            continue;
        }

        const size_t bit_index = first_free_bit_index_locked(candidate_words);
        const size_t word_index = (summary_word_index * 64) + bit_index;
        return word_index < word_count ? word_index : word_count;
    }

    return word_count;
}

bool MemoryPool::is_slot_allocated_locked(const Run& run, size_t slot_index) const {
    const size_t word_index = slot_index / 64;
    const size_t bit_index = slot_index % 64;
    return (run.bitmap[word_index] & (uint64_t{1} << bit_index)) != 0;
}

uint64_t MemoryPool::valid_mask_for_word_locked(const Run& run, size_t word_index) const {
    const size_t word_start = word_index * 64;
    if (word_start >= run.total_slots) {
        return 0;
    }

    const size_t remaining_bits = run.total_slots - word_start;
    if (remaining_bits >= 64) {
        return ~uint64_t{0};
    }
    return (uint64_t{1} << remaining_bits) - 1;
}

size_t MemoryPool::first_free_bit_index_locked(uint64_t free_bits) const {
    if (free_bits == 0) {
        return 64;
    }
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long bit_index = 0;
    _BitScanForward64(&bit_index, free_bits);
    return static_cast<size_t>(bit_index);
#elif defined(_MSC_VER)
    unsigned long bit_index = 0;
    _BitScanForward(&bit_index, static_cast<unsigned long>(free_bits & 0xffffffffULL));
    return static_cast<size_t>(bit_index);
#else
    return static_cast<size_t>(__builtin_ctzll(free_bits));
#endif
}

void MemoryPool::mark_slot_allocated_locked(Run& run, size_t slot_index) {
    const size_t word_index = slot_index / 64;
    const size_t bit_index = slot_index % 64;
    run.bitmap[word_index] |= (uint64_t{1} << bit_index);
    const uint64_t valid_mask = valid_mask_for_word_locked(run, word_index);
    set_summary_bit_locked(run, word_index, (run.bitmap[word_index] & valid_mask) != valid_mask);
}

void MemoryPool::mark_slot_free_locked(Run& run, size_t slot_index) {
    const size_t word_index = slot_index / 64;
    const size_t bit_index = slot_index % 64;
    run.bitmap[word_index] &= ~(uint64_t{1} << bit_index);
    set_summary_bit_locked(run, word_index, true);
}

void MemoryPool::set_summary_bit_locked(Run& run, size_t word_index, bool has_free_slot) {
    const size_t summary_word_index = word_index / 64;
    const size_t summary_bit_index = word_index % 64;
    if (has_free_slot) {
        run.summary_bitmap[summary_word_index] |= (uint64_t{1} << summary_bit_index);
    } else {
        run.summary_bitmap[summary_word_index] &= ~(uint64_t{1} << summary_bit_index);
    }
}

void MemoryPool::add_run_to_partial_locked(Run* run) {
    if (run == nullptr || run->owner == nullptr || run->owner_partial_bucket_index != static_cast<size_t>(-1)) {
        return;
    }

    if (run->free_slots == 0 || run->free_slots >= run->total_slots) {
        return;
    }

    auto& buckets = run->owner->partial_runs_by_free_slots;
    if (buckets.size() < run->total_slots + 1) {
        buckets.resize(run->total_slots + 1);
    }

    auto& bucket = buckets[run->free_slots];
    run->owner_partial_free_slots = run->free_slots;
    run->owner_partial_bucket_index = bucket.size();
    bucket.push_back(run);
    ++run->owner->partial_run_count;
}

void MemoryPool::remove_run_from_partial_locked(Run* run) {
    if (run == nullptr || run->owner == nullptr || run->owner_partial_bucket_index == static_cast<size_t>(-1)) {
        return;
    }

    auto& buckets = run->owner->partial_runs_by_free_slots;
    const size_t bucket_index = run->owner_partial_free_slots;
    auto& bucket = buckets[bucket_index];
    const size_t remove_index = run->owner_partial_bucket_index;
    Run* last = bucket.back();
    if (last != run) {
        bucket[remove_index] = last;
        last->owner_partial_bucket_index = remove_index;
    }
    bucket.pop_back();
    --run->owner->partial_run_count;
    run->owner_partial_bucket_index = static_cast<size_t>(-1);
    run->owner_partial_free_slots = static_cast<size_t>(-1);
}

void MemoryPool::add_run_to_empty_locked(Run* run) {
    if (run == nullptr || run->owner == nullptr || run->owner_empty_index != static_cast<size_t>(-1)) {
        return;
    }

    auto& runs = run->owner->empty_runs;
    run->owner_empty_index = runs.size();
    runs.push_back(run);
}

void MemoryPool::remove_run_from_empty_locked(Run* run) {
    if (run == nullptr || run->owner == nullptr || run->owner_empty_index == static_cast<size_t>(-1)) {
        return;
    }

    auto& runs = run->owner->empty_runs;
    const size_t remove_index = run->owner_empty_index;
    Run* last = runs.back();
    if (last != run) {
        runs[remove_index] = last;
        last->owner_empty_index = remove_index;
    }
    runs.pop_back();
    run->owner_empty_index = static_cast<size_t>(-1);
}

void MemoryPool::add_run_to_class_locked(DomainState& domain, Run* run) {
    if (run == nullptr || run->class_runs_index != static_cast<size_t>(-1)) {
        return;
    }

    auto& runs = domain.runs_by_class[run->slot_size];
    run->class_runs_index = runs.size();
    runs.push_back(run);
}

void MemoryPool::remove_run_from_class_locked(DomainState& domain, Run* run) {
    if (run == nullptr || run->class_runs_index == static_cast<size_t>(-1)) {
        return;
    }

    auto it = domain.runs_by_class.find(run->slot_size);
    if (it == domain.runs_by_class.end()) {
        run->class_runs_index = static_cast<size_t>(-1);
        return;
    }

    auto& runs = it->second;
    const size_t remove_index = run->class_runs_index;
    Run* last = runs.back();
    if (last != run) {
        runs[remove_index] = last;
        last->class_runs_index = remove_index;
    }
    runs.pop_back();
    run->class_runs_index = static_cast<size_t>(-1);
    if (runs.empty()) {
        domain.runs_by_class.erase(it);
    }
}

void MemoryPool::update_run_state_after_alloc_locked(Run* run, RunState previous_state) {
    if (previous_state == RunState::Empty) {
        remove_run_from_empty_locked(run);
    } else if (previous_state == RunState::Partial) {
        remove_run_from_partial_locked(run);
    }

    if (run->free_slots == 0) {
        run->state = RunState::Full;
        return;
    }

    run->state = RunState::Partial;
    add_run_to_partial_locked(run);
}

void MemoryPool::update_run_state_after_free_locked(Run* run, RunState previous_state) {
    if (previous_state == RunState::Partial) {
        remove_run_from_partial_locked(run);
    }

    if (run->free_slots == run->total_slots) {
        run->state = RunState::Empty;
        if (previous_state != RunState::Empty) {
            add_run_to_empty_locked(run);
        }
        return;
    }

    run->state = RunState::Partial;
    add_run_to_partial_locked(run);
}

void MemoryPool::update_large_block_state_counters(DomainState& domain, LargeBlockState state, size_t bytes, bool add) {
    size_t* counter = nullptr;
    switch (state) {
    case LargeBlockState::Dirty:
        counter = &domain.stats.dirty_large_bytes;
        break;
    case LargeBlockState::Retained:
        counter = &domain.stats.retained_large_bytes;
        break;
    case LargeBlockState::Active:
        return;
    }

    if (add) {
        *counter += bytes;
    } else {
        *counter -= bytes;
    }
}

MemoryPool::LargeBlockState MemoryPool::large_block_state(const Block& block) const {
    return block.in_free_list ? block.free_state : LargeBlockState::Active;
}

size_t MemoryPool::current_total_active_bytes_locked() const {
    return tensor_domain_.stats.active_bytes + workspace_domain_.stats.active_bytes;
}

void MemoryPool::update_total_peak_active_locked() {
    peak_active_bytes_total_ = std::max(peak_active_bytes_total_, current_total_active_bytes_locked());
}

} // namespace memory
} // namespace asnumpy
