import asnumpy as ap
import numpy as np

print("--- Step 1: Alloc 8MB ---")
# 申请一个大块 (8MB)
a = ap.zeros((2000, 1000), dtype=np.float32) # 8MB
ptr_a = a.device_ptr
print(f"Allocated A (8MB) at {ptr_a}")

print("\n--- Step 2: Free 8MB ---")
del a 
# 此时，池子里应该有一个 8MB 的空闲块

print("\n--- Step 3: Alloc 4MB (Should SPLIT) ---")
# 申请一个小块 (4MB)
b = ap.zeros((1000, 1000), dtype=np.float32) # 4MB
ptr_b = b.device_ptr
print(f"Allocated B (4MB) at {ptr_b}")

# 预期：ptr_b 应该等于 ptr_a（复用了头部），且日志显示 SPLIT

print("\n--- Step 4: Alloc another 4MB (Should use Remainder) ---")
c = ap.zeros((1000, 1000), dtype=np.float32) # 4MB
ptr_c = c.device_ptr
print(f"Allocated C (4MB) at {ptr_c}")

# 预期：ptr_c 应该等于 ptr_a + 4MB，日志显示 REUSE