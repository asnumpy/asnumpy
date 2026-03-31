#include "asnumpy/memory/MemoryPool.hpp"
#include <cstdlib> // 用于 std::getenv
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

namespace asnumpy {
namespace memory {

constexpr size_t ALIGNMENT = 512;
constexpr size_t MIN_SPLIT_SIZE = 1024;

// 辅助函数：检查是否开启调试日志
bool is_debug_mode() {
    static bool inited = false;
    static bool debug = false;
    if (!inited) {
        const char* env = std::getenv("ASN_DEBUG_LOG");
        if (env != nullptr && std::string(env) == "1") {
            debug = true;
        }
        inited = true;
    }
    return debug;
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool instance;
    return instance;
}

MemoryPool::~MemoryPool() {
    clear_cache();
}

size_t round_up(size_t size) {
    return (size + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
}

void* MemoryPool::malloc(size_t size) {
    // 【开关1】对比测试：ASN_ENABLE_POOL=0 -> 走系统调用
    const char* env_p = std::getenv("ASN_ENABLE_POOL");
    if (env_p != nullptr && std::string(env_p) == "0") {
        void* ptr = nullptr;
        auto ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("System aclrtMalloc failed");
        }
        return ptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    size_t alloc_size = round_up(size);

    auto it = free_blocks_.lower_bound(alloc_size);

    if (it != free_blocks_.end()) {
        Block* block = it->second;
        remove_from_free_list(block);

        if (block->size >= alloc_size + MIN_SPLIT_SIZE) {
            size_t remaining_size = block->size - alloc_size;
            void* remaining_ptr = static_cast<char*>(block->ptr) + alloc_size;

            Block* remaining_block = new Block(remaining_ptr, remaining_size, false, false);
            remaining_block->prev = block;
            remaining_block->next = block->next;
            if (remaining_block->next) remaining_block->next->prev = remaining_block;
            block->next = remaining_block;
            block->size = alloc_size;

            all_blocks_[remaining_ptr] = remaining_block;
            add_to_free_list(remaining_block);

            // 【开关2】调试日志
            if (is_debug_mode()) {
                std::cerr << "!!! [MemoryPool] SPLIT: Block -> " << alloc_size << " (Use) + " << remaining_size << " (Free) !!!" << std::endl;
            }
        } else {
            if (is_debug_mode()) {
                std::cerr << "!!! [MemoryPool] REUSE: Exact/Near fit " << block->size << " for " << alloc_size << " !!!" << std::endl;
            }
        }

        block->allocated = true;
        return block->ptr;
    }

    size_t system_alloc_size = (alloc_size > kDefaultArenaSize) ? alloc_size : kDefaultArenaSize;
    void* ptr = nullptr;
    auto ret = aclrtMalloc(&ptr, system_alloc_size, ACL_MEM_MALLOC_HUGE_FIRST);

    if (ret != ACL_SUCCESS) {
        std::string msg = "MemoryPool: aclrtMalloc failed ret=" + std::to_string(ret);
        throw std::runtime_error(msg);
    }

    Block* block = new Block(ptr, system_alloc_size, false, true);
    all_blocks_[ptr] = block;

    if (is_debug_mode()) {
        std::cerr << "!!! [MemoryPool] ARENA ALLOC: System allocated " << system_alloc_size << " bytes at " << ptr << " !!!" << std::endl;
    }

    add_to_free_list(block);
    remove_from_free_list(block);

    if (block->size >= alloc_size + MIN_SPLIT_SIZE) {
        size_t remaining_size = block->size - alloc_size;
        void* remaining_ptr = static_cast<char*>(block->ptr) + alloc_size;

        Block* remaining_block = new Block(remaining_ptr, remaining_size, false, false);
        remaining_block->prev = block;
        remaining_block->next = block->next; 
        block->next = remaining_block;
        block->size = alloc_size;

        all_blocks_[remaining_ptr] = remaining_block;
        add_to_free_list(remaining_block);

        if (is_debug_mode()) {
            std::cerr << "!!! [MemoryPool] ARENA SPLIT: " << alloc_size << " (Use) + " << remaining_size << " (Free) !!!" << std::endl;
        }
    }

    block->allocated = true;
    return block->ptr;
}

void MemoryPool::free(void* ptr) {
    if (ptr == nullptr) return;

    // 【开关1】对比测试
    const char* env_p = std::getenv("ASN_ENABLE_POOL");
    if (env_p != nullptr && std::string(env_p) == "0") {
        aclrtFree(ptr);
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = all_blocks_.find(ptr);
    if (it == all_blocks_.end()) {
        if (is_debug_mode()) std::cerr << "[MemoryPool] Error: Unknown pointer " << ptr << std::endl;
        return;
    }

    Block* block = it->second;
    block->allocated = false;
    block = try_merge(block);
    add_to_free_list(block);

    // 【开关2】调试日志
    if (is_debug_mode()) {
        std::cerr << "!!! [MemoryPool] FREE -> Cache (Size: " << block->size << ") !!!" << std::endl;
    }
}

Block* MemoryPool::try_merge(Block* block) {
    if (block->next && !block->next->allocated) {
        Block* next_block = block->next;
        remove_from_free_list(next_block);

        block->size += next_block->size;
        block->next = next_block->next;
        if (block->next) block->next->prev = block;

        all_blocks_.erase(next_block->ptr);
        delete next_block;
        
        if (is_debug_mode()) std::cerr << "    -> Merged NEXT" << std::endl;
    }

    if (block->prev && !block->prev->allocated) {
        Block* prev_block = block->prev;
        remove_from_free_list(prev_block);

        prev_block->size += block->size;
        prev_block->next = block->next;
        if (prev_block->next) prev_block->next->prev = prev_block;

        all_blocks_.erase(block->ptr);
        delete block;
        
        if (is_debug_mode()) std::cerr << "    -> Merged PREV" << std::endl;
        return prev_block;
    }
    return block;
}

void MemoryPool::add_to_free_list(Block* block) {
    free_blocks_.insert({block->size, block});
}

void MemoryPool::remove_from_free_list(Block* block) {
    auto range = free_blocks_.equal_range(block->size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second == block) {
            free_blocks_.erase(it);
            return;
        }
    }
}

void MemoryPool::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t freed_arenas = 0;
    size_t freed_bytes = 0;

    std::vector<Block*> heads_to_free;

    for (auto& pair : all_blocks_) {
        Block* b = pair.second;
        if (b->is_head && !b->allocated && b->next == nullptr) {
            heads_to_free.push_back(b);
        }
    }

    for (Block* head : heads_to_free) {
        remove_from_free_list(head);
        all_blocks_.erase(head->ptr);
        aclrtFree(head->ptr);
        freed_bytes += head->size;
        freed_arenas++;
        delete head;
    }

    if (freed_arenas > 0 && is_debug_mode()) {
        std::cerr << "[MemoryPool] Cache cleared: " << freed_arenas << " arenas (" << freed_bytes << " bytes)" << std::endl;
    }
}

} // namespace memory
} // namespace asnumpy