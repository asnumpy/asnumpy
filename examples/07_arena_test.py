import asnumpy as ap
import numpy as np

print("\n--- 1. Alloc Small 4KB (Trigger Arena 64MB) ---")
# 这应该触发一次 64MB 的系统申请，然后切出 4KB 给 A
a = ap.zeros(1024, dtype=np.float32) 
print(f"Array A ptr: {a.device_ptr}")

print("\n--- 2. Alloc Another 4MB (Reuse Arena) ---")
# 这应该直接在刚才的 64MB 里切，不会有 System alloc
# 只需要切分日志 (SPLIT)
b = ap.zeros((1000, 1000), dtype=np.float32) 
print(f"Array B ptr: {b.device_ptr}")

print("\n--- 3. Free Both ---")
del b # B 释放，此时还没法完全还给系统，因为 A 还在用
del a # A 释放，A 和 B 合并回 64MB 大块

print("\n--- 4. End ---")
# 程序结束时，MemoryPool 析构，应该打印 Cache cleared