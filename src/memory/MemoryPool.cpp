#include "asnumpy/memory/MemoryPool.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace asnumpy {
namespace memory {

namespace {

constexpr size_t kAlignment = 512;
constexpr size_t kSmallAllocationMax = 1 * 1024 * 1024;
constexpr size_t kSmallRunTargetSize = 2 * 1024 * 1024;
constexpr size_t kSmallRunMinSlots = 4;
constexpr size_t kSmallRunMaxSlots = 128;
constexpr size_t kLargeMinSplitSize = 4 * 1024;

constexpr std::array<size_t, 12> kSmallSizeClasses = {
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
};

bool is_debug_mode() {
    static bool inited = false;
    static bool debug = false;
    if (!inited) {
        const char* env = std::getenv("ASN_DEBUG_LOG");
        debug = env != nullptr && std::string(env) == "1";
        inited = true;
    }
    return debug;
}

bool is_pool_enabled() {
    const char* env = std::getenv("ASN_ENABLE_POOL");
    return env == nullptr || std::string(env) != "0";
}

} // namespace

MemoryPool& MemoryPool::instance() {
    static MemoryPool instance;
    return instance;
}

MemoryPool::~MemoryPool() {
    clear_cache();
}

size_t MemoryPool::round_up(size_t size) const {
    return ((size + kAlignment - 1) / kAlignment) * kAlignment;
}

size_t MemoryPool::size_class_for(size_t size) const {
    if (size == 0 || size > kSmallAllocationMax) {
        return 0;
    }
    for (size_t size_class : kSmallSizeClasses) {
        if (size <= size_class) {
            return size_class;
        }
    }
    return 0;
}

void* MemoryPool::malloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    if (!is_pool_enabled()) {
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("System aclrtMalloc failed");
        }
        return ptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const size_t alloc_size = round_up(size);
    ++stats_.allocation_requests;

    const size_t size_class = size_class_for(alloc_size);
    if (size_class != 0) {
        return allocate_small_locked(size_class);
    }
    return allocate_large_locked(alloc_size);
}

void* MemoryPool::allocate_small_locked(size_t alloc_size) {
    auto& bin = small_bins_[alloc_size];
    bool had_cached_block = !bin.empty();
    if (bin.empty()) {
        allocate_run_locked(alloc_size);
    }

    Block* block = bin.back();
    bin.pop_back();
    block->allocated = true;
    --block->run->free_slots;

    if (had_cached_block) {
        ++stats_.cache_hits;
    }
    stats_.cached_bytes -= block->size;
    stats_.active_bytes += block->size;
    stats_.peak_active_bytes = std::max(stats_.peak_active_bytes, stats_.active_bytes);

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] SMALL ALLOC class=" << alloc_size
                  << " ptr=" << block->ptr
                  << " free_slots=" << block->run->free_slots
                  << "/" << block->run->total_slots << std::endl;
    }
    return block->ptr;
}

MemoryPool::Run* MemoryPool::allocate_run_locked(size_t slot_size) {
    size_t target_slots = std::max<size_t>(1, kSmallRunTargetSize / slot_size);
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

    auto& bin = small_bins_[slot_size];
    for (size_t i = 0; i < slot_count; ++i) {
        void* slot_ptr = static_cast<char*>(ptr) + (i * slot_size);
        Block* block = new Block(slot_ptr, slot_size, false);
        block->is_small = true;
        block->run = run;
        run->blocks.push_back(block);
        all_blocks_[slot_ptr] = block;
        bin.push_back(block);
    }

    runs_by_class_[slot_size].push_back(run);
    ++stats_.cache_misses;
    ++stats_.system_allocations;
    ++stats_.small_run_count;
    stats_.system_bytes += run_size;
    stats_.cached_bytes += run_size;

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] SMALL RUN alloc slot_size=" << slot_size
                  << " run_size=" << run_size
                  << " slots=" << slot_count
                  << " ptr=" << ptr << std::endl;
    }
    return run;
}

void* MemoryPool::allocate_large_locked(size_t alloc_size) {
    auto it = large_free_blocks_.lower_bound(alloc_size);
    Block* block = nullptr;
    bool from_cache = false;

    if (it != large_free_blocks_.end()) {
        block = it->second;
        remove_from_large_free_list(block);
        from_cache = true;
        ++stats_.cache_hits;
        stats_.cached_bytes -= block->size;
    } else {
        const size_t system_alloc_size =
            alloc_size > kDefaultArenaSize ? alloc_size : kDefaultArenaSize;
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, system_alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("MemoryPool: aclrtMalloc failed ret=" + std::to_string(ret));
        }

        block = new Block(ptr, system_alloc_size, false);
        block->is_head = true;
        all_blocks_[ptr] = block;

        ++stats_.cache_misses;
        ++stats_.system_allocations;
        ++stats_.large_arena_count;
        stats_.system_bytes += system_alloc_size;

        if (is_debug_mode()) {
            std::cerr << "[MemoryPool] LARGE arena alloc size=" << system_alloc_size
                      << " ptr=" << ptr << std::endl;
        }
    }

    if (block->size >= alloc_size + kLargeMinSplitSize) {
        const size_t remaining_size = block->size - alloc_size;
        void* remaining_ptr = static_cast<char*>(block->ptr) + alloc_size;

        Block* remaining = new Block(remaining_ptr, remaining_size, false);
        remaining->prev = block;
        remaining->next = block->next;
        if (remaining->next != nullptr) {
            remaining->next->prev = remaining;
        }
        block->next = remaining;
        block->size = alloc_size;

        all_blocks_[remaining_ptr] = remaining;
        add_to_large_free_list(remaining);
        stats_.cached_bytes += remaining_size;
    }

    block->allocated = true;
    stats_.active_bytes += block->size;
    stats_.peak_active_bytes = std::max(stats_.peak_active_bytes, stats_.active_bytes);

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] LARGE "
                  << (from_cache ? "reuse" : "direct")
                  << " size=" << block->size
                  << " ptr=" << block->ptr << std::endl;
    }
    return block->ptr;
}

void MemoryPool::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }

    if (!is_pool_enabled()) {
        aclrtFree(ptr);
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = all_blocks_.find(ptr);
    if (it == all_blocks_.end()) {
        if (is_debug_mode()) {
            std::cerr << "[MemoryPool] Unknown pointer " << ptr << std::endl;
        }
        return;
    }

    ++stats_.free_requests;
    Block* block = it->second;
    if (!block->allocated) {
        if (is_debug_mode()) {
            std::cerr << "[MemoryPool] Double free ignored ptr=" << ptr << std::endl;
        }
        return;
    }

    if (block->is_small) {
        free_small_locked(block);
        return;
    }
    free_large_locked(block);
}

void MemoryPool::free_small_locked(Block* block) {
    block->allocated = false;
    ++block->run->free_slots;
    small_bins_[block->size].push_back(block);

    stats_.active_bytes -= block->size;
    stats_.cached_bytes += block->size;

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] SMALL FREE class=" << block->size
                  << " ptr=" << block->ptr
                  << " free_slots=" << block->run->free_slots
                  << "/" << block->run->total_slots << std::endl;
    }
}

void MemoryPool::free_large_locked(Block* block) {
    const size_t released_size = block->size;
    block->allocated = false;
    Block* merged = try_merge_locked(block);
    add_to_large_free_list(merged);

    stats_.active_bytes -= released_size;
    stats_.cached_bytes += released_size;

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] LARGE FREE size=" << released_size
                  << " cached_as=" << merged->size
                  << " ptr=" << merged->ptr << std::endl;
    }
}

MemoryPool::Block* MemoryPool::try_merge_locked(Block* block) {
    if (block->next != nullptr && !block->next->allocated && !block->next->is_small) {
        Block* next_block = block->next;
        remove_from_large_free_list(next_block);

        block->size += next_block->size;
        block->next = next_block->next;
        if (block->next != nullptr) {
            block->next->prev = block;
        }

        all_blocks_.erase(next_block->ptr);
        delete next_block;
    }

    if (block->prev != nullptr && !block->prev->allocated && !block->prev->is_small) {
        Block* prev_block = block->prev;
        remove_from_large_free_list(prev_block);

        prev_block->size += block->size;
        prev_block->next = block->next;
        if (prev_block->next != nullptr) {
            prev_block->next->prev = prev_block;
        }

        all_blocks_.erase(block->ptr);
        delete block;
        return prev_block;
    }

    return block;
}

void MemoryPool::add_to_large_free_list(Block* block) {
    large_free_blocks_.insert({block->size, block});
}

void MemoryPool::remove_from_large_free_list(Block* block) {
    auto range = large_free_blocks_.equal_range(block->size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == block) {
            large_free_blocks_.erase(it);
            return;
        }
    }
}

void MemoryPool::release_run_locked(Run* run) {
    auto& bin = small_bins_[run->slot_size];
    for (Block* block : run->blocks) {
        bin.erase(std::remove(bin.begin(), bin.end(), block), bin.end());
        all_blocks_.erase(block->ptr);
        delete block;
    }

    auto& runs = runs_by_class_[run->slot_size];
    runs.erase(std::remove(runs.begin(), runs.end(), run), runs.end());

    aclrtFree(run->ptr);
    ++stats_.system_frees;
    --stats_.small_run_count;
    stats_.system_bytes -= run->size;
    stats_.cached_bytes -= run->size;

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] SMALL RUN release slot_size=" << run->slot_size
                  << " run_size=" << run->size
                  << " ptr=" << run->ptr << std::endl;
    }
    delete run;
}

void MemoryPool::release_large_head_locked(Block* block) {
    remove_from_large_free_list(block);
    aclrtFree(block->ptr);
    all_blocks_.erase(block->ptr);

    ++stats_.system_frees;
    --stats_.large_arena_count;
    stats_.system_bytes -= block->size;
    stats_.cached_bytes -= block->size;

    if (is_debug_mode()) {
        std::cerr << "[MemoryPool] LARGE arena release size=" << block->size
                  << " ptr=" << block->ptr << std::endl;
    }
    delete block;
}

void MemoryPool::clear_cache() {
    if (!is_pool_enabled()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Run*> releasable_runs;
    for (auto& entry : runs_by_class_) {
        for (Run* run : entry.second) {
            if (run->free_slots == run->total_slots) {
                releasable_runs.push_back(run);
            }
        }
    }

    for (Run* run : releasable_runs) {
        release_run_locked(run);
    }

    std::vector<Block*> releasable_heads;
    for (auto& entry : all_blocks_) {
        Block* block = entry.second;
        if (!block->is_small && block->is_head && !block->allocated && block->next == nullptr) {
            releasable_heads.push_back(block);
        }
    }

    for (Block* head : releasable_heads) {
        release_large_head_locked(head);
    }
}

MemoryPoolStats MemoryPool::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

} // namespace memory
} // namespace asnumpy
