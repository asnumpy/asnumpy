/**
 * Memory pool for CANN device allocations.
 *
 * The pool now separates tensor payloads and operator workspaces into
 * independent domains so their caching and trimming policies can evolve
 * independently without changing Python-level array construction semantics.
 */
#pragma once

#include <acl/acl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace asnumpy {
namespace memory {

enum class PoolDomain {
    Tensor,
    Workspace,
};

struct MemoryPoolStats {
    size_t allocation_requests = 0;
    size_t free_requests = 0;
    size_t cache_hits = 0;
    size_t cache_misses = 0;
    size_t small_allocation_requests = 0;
    size_t small_cache_hits = 0;
    size_t small_cache_misses = 0;
    size_t large_allocation_requests = 0;
    size_t large_cache_hits = 0;
    size_t large_cache_misses = 0;
    size_t system_allocations = 0;
    size_t system_frees = 0;
    size_t active_bytes = 0;
    size_t active_requested_bytes = 0;
    size_t cached_bytes = 0;
    size_t system_bytes = 0;
    size_t peak_active_bytes = 0;
    size_t small_run_count = 0;
    size_t large_arena_count = 0;
    size_t empty_small_run_count = 0;
    size_t partial_small_run_count = 0;
    size_t full_small_run_count = 0;
    size_t small_slots_total = 0;
    size_t small_slots_free = 0;
    size_t bitmap_words_scanned = 0;
    size_t summary_words_scanned = 0;
    size_t dirty_large_block_count = 0;
    size_t retained_large_block_count = 0;
    size_t dirty_large_bytes = 0;
    size_t retained_large_bytes = 0;
    size_t large_split_count = 0;
    size_t large_merge_count = 0;
    size_t large_decay_count = 0;
    size_t trimmed_bytes = 0;
    size_t small_free_bytes = 0;
    size_t largest_free_run_bytes = 0;
    size_t largest_free_block_bytes = 0;
    size_t internal_fragmentation_bytes = 0;
    size_t small_external_fragmentation_bytes = 0;
    size_t large_external_fragmentation_bytes = 0;
    size_t external_fragmentation_bytes = 0;
    size_t hot_retained_empty_runs = 0;
    double internal_fragmentation_ratio_pct = 0.0;
    double small_external_fragmentation_ratio_pct = 0.0;
    double large_external_fragmentation_ratio_pct = 0.0;
    double external_fragmentation_ratio_pct = 0.0;
    double run_utilization_pct = 0.0;
};

class MemoryPool {
public:
    static MemoryPool& instance();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void* malloc(size_t size);
    void* malloc(size_t size, PoolDomain domain);

    void free(void* ptr);
    void free(void* ptr, PoolDomain domain);

    void clear_cache();
    void clear_cache(PoolDomain domain);

    void trim();
    void trim(PoolDomain domain);

    void refresh_config();

    void reset_stats();
    void reset_stats(PoolDomain domain);

    MemoryPoolStats stats() const;
    MemoryPoolStats stats(PoolDomain domain) const;

private:
    struct Bin;
    struct Run;
    struct Block;
    using LargeFreeList = std::multimap<size_t, Block*>;

    struct Block {
        void* ptr = nullptr;
        size_t size = 0;
        bool allocated = false;
        bool is_head = false;
        uint64_t last_used_epoch = 0;
        uint64_t decay_epoch = 0;
        Block* prev = nullptr;
        Block* next = nullptr;
        LargeFreeList::iterator free_list_iter{};
        bool in_free_list = false;

        Block() = default;
        Block(void* block_ptr, size_t block_size, bool in_use)
            : ptr(block_ptr), size(block_size), allocated(in_use) {}
    };

    enum class RunState {
        Empty,
        Partial,
        Full,
    };

    enum class LargeBlockState {
        Active,
        Dirty,
        Retained,
    };

    struct Run {
        void* ptr = nullptr;
        size_t size = 0;
        size_t slot_size = 0;
        size_t total_slots = 0;
        size_t free_slots = 0;
        size_t first_free_hint = 0;
        std::vector<uint64_t> bitmap;
        std::vector<uint64_t> summary_bitmap;
        Bin* owner = nullptr;
        RunState state = RunState::Empty;
        bool hot_retained = false;
        size_t owner_partial_bucket_index = static_cast<size_t>(-1);
        size_t owner_partial_free_slots = static_cast<size_t>(-1);
        size_t owner_empty_index = static_cast<size_t>(-1);
        size_t class_runs_index = static_cast<size_t>(-1);
    };

    struct Bin {
        size_t size_class = 0;
        std::vector<std::vector<Run*>> partial_runs_by_free_slots;
        size_t partial_run_count = 0;
        std::vector<Run*> empty_runs;
        size_t alloc_count = 0;
        size_t last_trim_alloc_count = 0;
        size_t idle_generations = 0;
    };

    struct DomainConfig {
        size_t small_run_target_size = 0;
        uint64_t large_dirty_decay_epochs = 0;
        bool aggressive_trim = false;
        size_t hot_bin_min_allocations = 0;
        size_t hot_bin_max_empty_runs = 0;
        size_t hot_bin_max_idle_generations = 0;
    };

    struct DomainState {
        std::unordered_map<void*, Block*> all_blocks;
        std::unordered_map<void*, size_t> active_requested_sizes;
        std::unordered_map<size_t, Bin> small_bins;
        std::unordered_map<size_t, std::vector<Run*>> runs_by_class;
        std::map<uintptr_t, Run*> small_runs_by_base;
        LargeFreeList dirty_large_blocks;
        LargeFreeList retained_large_blocks;
        std::map<uint64_t, std::vector<void*>> dirty_large_decay_buckets;
        std::unordered_map<Block*, LargeBlockState> large_block_states;
        uint64_t large_epoch = 0;
        MemoryPoolStats stats;
    };

    static constexpr size_t kDefaultArenaSize = 64 * 1024 * 1024;

    MemoryPool();
    ~MemoryPool();

    DomainState& domain_state(PoolDomain domain);
    const DomainState& domain_state(PoolDomain domain) const;
    const DomainConfig& domain_config(PoolDomain domain) const;

    void* allocate_small_locked(DomainState& domain, const DomainConfig& config, size_t alloc_size, size_t requested_size);
    void* allocate_large_locked(DomainState& domain, const DomainConfig& config, size_t alloc_size, size_t requested_size);
    void free_small_locked(DomainState& domain, Run* run, size_t slot_index, void* ptr, size_t requested_size);
    void free_large_locked(DomainState& domain, const DomainConfig& config, Block* block, size_t requested_size);
    bool free_in_domain_locked(DomainState& domain, const DomainConfig& config, void* ptr);

    size_t round_up(size_t size) const;
    size_t size_class_for(size_t size) const;
    Run* allocate_run_locked(DomainState& domain, const DomainConfig& config, size_t slot_size);
    void release_run_locked(DomainState& domain, Run* run);
    Run* select_fullest_partial_run_locked(Bin& bin);

    void trim_locked(DomainState& domain, const DomainConfig& config, bool release_all_large);
    void decay_large_blocks_locked(DomainState& domain, const DomainConfig& config, bool force_all);
    void refresh_config_locked();

    Run* find_small_run_locked(const DomainState& domain, void* ptr) const;
    size_t find_free_slot_locked(DomainState& domain, const Run& run);
    size_t find_free_word_locked(DomainState& domain, const Run& run, size_t start_word);
    bool is_slot_allocated_locked(const Run& run, size_t slot_index) const;
    uint64_t valid_mask_for_word_locked(const Run& run, size_t word_index) const;
    size_t first_free_bit_index_locked(uint64_t free_bits) const;
    void mark_slot_allocated_locked(Run& run, size_t slot_index);
    void mark_slot_free_locked(Run& run, size_t slot_index);
    void set_summary_bit_locked(Run& run, size_t word_index, bool has_free_slot);
    void add_run_to_partial_locked(Run* run);
    void remove_run_from_partial_locked(Run* run);
    void add_run_to_empty_locked(Run* run);
    void remove_run_from_empty_locked(Run* run);
    void add_run_to_class_locked(DomainState& domain, Run* run);
    void remove_run_from_class_locked(DomainState& domain, Run* run);
    void update_run_state_after_alloc_locked(Run* run, RunState previous_state);
    void update_run_state_after_free_locked(Run* run, RunState previous_state);

    void add_to_large_free_list(DomainState& domain, const DomainConfig& config, Block* block, LargeBlockState state);
    void remove_from_large_free_list(DomainState& domain, Block* block);
    Block* try_merge_locked(DomainState& domain, Block* block);
    void release_large_head_locked(DomainState& domain, Block* block);
    void update_large_block_state_counters(DomainState& domain, LargeBlockState state, size_t bytes, bool add);
    LargeBlockState large_block_state(const DomainState& domain, const Block& block) const;
    size_t current_total_active_bytes_locked() const;
    void update_total_peak_active_locked();

    MemoryPoolStats snapshot_domain_stats_locked(const DomainState& domain) const;
    void reset_domain_stats_locked(DomainState& domain);
    static void accumulate_stats(MemoryPoolStats& dst, const MemoryPoolStats& src);

    mutable std::mutex mutex_;
    std::atomic<bool> pool_enabled_{true};
    DomainState tensor_domain_;
    DomainState workspace_domain_;
    DomainConfig tensor_config_;
    DomainConfig workspace_config_;
    size_t peak_active_bytes_total_ = 0;
};

} // namespace memory
} // namespace asnumpy
