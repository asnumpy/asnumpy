/**
 * 文件: include/asnumpy/memory/MemoryPool.hpp
 * 说明: 引入 Arena 机制和 Block Head 标记
 */
 #pragma once

 #include <cstddef>
 #include <mutex>
 #include <iostream>
 #include <map>
 #include <unordered_map>
 #include <vector>
 #include <acl/acl.h>
 
 namespace asnumpy {
    namespace memory {
    
    struct MemoryPoolStats {
        size_t allocation_requests = 0;
        size_t free_requests = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t system_allocations = 0;
        size_t system_frees = 0;
        size_t active_bytes = 0;
        size_t cached_bytes = 0;
        size_t system_bytes = 0;
        size_t peak_active_bytes = 0;
        size_t small_run_count = 0;
        size_t large_arena_count = 0;
    };
    
    class MemoryPool {
    public:
        static MemoryPool& instance();
    
        MemoryPool(const MemoryPool&) = delete;
        MemoryPool& operator=(const MemoryPool&) = delete;
    
        void* malloc(size_t size);
        void free(void* ptr);
        void clear_cache();
        MemoryPoolStats stats() const;
    
    private:
        struct Run;
    
        struct Block {
            void* ptr = nullptr;
            size_t size = 0;
            bool allocated = false;
            bool is_small = false;
            bool is_head = false;
            Run* run = nullptr;
            Block* prev = nullptr;
            Block* next = nullptr;
    
            Block() = default;
            Block(void* block_ptr, size_t block_size, bool in_use)
                : ptr(block_ptr), size(block_size), allocated(in_use) {}
        };
    
        struct Run {
            void* ptr = nullptr;
            size_t size = 0;
            size_t slot_size = 0;
            size_t total_slots = 0;
            size_t free_slots = 0;
            std::vector<Block*> blocks;
        };
    
        static constexpr size_t kDefaultArenaSize = 64 * 1024 * 1024;
    
        MemoryPool() = default;
        ~MemoryPool();
    
        void* allocate_small_locked(size_t alloc_size);
        void* allocate_large_locked(size_t alloc_size);
        void free_small_locked(Block* block);
        void free_large_locked(Block* block);
    
        size_t round_up(size_t size) const;
        size_t size_class_for(size_t size) const;
        Run* allocate_run_locked(size_t slot_size);
        void release_run_locked(Run* run);
    
        void add_to_large_free_list(Block* block);
        void remove_from_large_free_list(Block* block);
        Block* try_merge_locked(Block* block);
        void release_large_head_locked(Block* block);
    
        mutable std::mutex mutex_;
        std::unordered_map<void*, Block*> all_blocks_;
        std::unordered_map<size_t, std::vector<Block*>> small_bins_;
        std::unordered_map<size_t, std::vector<Run*>> runs_by_class_;
        std::multimap<size_t, Block*> large_free_blocks_;
        MemoryPoolStats stats_;
    };
    
    } // namespace memory
    } // namespace asnumpy
    