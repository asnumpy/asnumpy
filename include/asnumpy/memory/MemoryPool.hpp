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
 
 // 默认 Arena 大小：64MB
 // 策略：只要申请小于 64MB，我们都直接找系统申请 64MB，剩下的留着慢慢切
 constexpr size_t kDefaultArenaSize = 64 * 1024 * 1024;
 
 struct Block {
     void* ptr;              // 物理地址
     size_t size;            // 当前块大小
     bool allocated;         // 是否被占用
     
     Block* prev = nullptr;  // 前驱 (物理相邻)
     Block* next = nullptr;  // 后继 (物理相邻)
     
     // [新增] 标记是否为系统分配的原始大块的头部
     // 只有 is_head=true 的块，在 clear_cache 时才能调用 aclrtFree
     bool is_head = false; 
 
     Block(void* p, size_t s, bool a, bool head=false) 
         : ptr(p), size(s), allocated(a), is_head(head) {}
 };
 
 class MemoryPool {
 public:
     static MemoryPool& instance();
 
     MemoryPool(const MemoryPool&) = delete;
     MemoryPool& operator=(const MemoryPool&) = delete;
 
     void* malloc(size_t size);
     void free(void* ptr);
     
     // 清理缓存：释放所有完全空闲的 Arena
     void clear_cache();
 
 private:
     MemoryPool() = default;
     ~MemoryPool();
 
     std::mutex mutex_;
 
     // 核心数据结构
     std::unordered_map<void*, Block*> all_blocks_;
     std::multimap<size_t, Block*> free_blocks_;
     
     void add_to_free_list(Block* block);
     void remove_from_free_list(Block* block);
     Block* try_merge(Block* block);
 };
 
 } // namespace memory
 } // namespace asnumpy