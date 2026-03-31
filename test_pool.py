import asnumpy as ap
import os

def test_memory_pool():
    print("\n" + "="*50)
    print("开始内存池验证测试...")
    print("="*50)

    # 步骤 1: 第一次创建数组
    # 预期日志：触发 [MemoryPool] ARENA ALLOC (申请大块) 
    #         和 [MemoryPool] ARENA SPLIT (切分给当前使用)
    print("\n--- 步骤 1: 创建 4MB 数组 'a' ---")
    a = ap.zeros((1024, 1024), dtype='float32')
    print("数组 'a' 已创建。")

    # 步骤 2: 释放数组
    # 预期日志：触发 !!! [MemoryPool] FREE -> Cache
    print("\n--- 步骤 2: 销毁数组 'a' ---")
    del a
    print("数组 'a' 已销毁，显存应回到池中。")

    # 步骤 3: 再次创建相同大小的数组
    # 预期逻辑：内存池会发现池子里刚好有一个 4MB 的空闲块
    # 预期日志：触发 !!! [MemoryPool] REUSE (复用) 
    #         注意：此时【不应该】再出现 ARENA ALLOC
    print("\n--- 步骤 3: 再次创建 4MB 数组 'b' ---")
    b = ap.zeros((1024, 1024), dtype='float32')
    print("数组 'b' 已创建。")

    print("\n" + "="*50)
    print("测试指令已发完，请检查上方日志中的 REUSE 标记！")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_memory_pool()