import asnumpy as ap
import numpy as np
import time
import os

def run_benchmark():
    # 这里的 shape 设为 1MB 左右，足以体现调度开销
    shape = (256 * 1024,) 
    iterations = 5000
    
    # 获取当前模式名称
    mode = "System Call (aclrtMalloc)" if os.environ.get("ASN_ENABLE_POOL") == "0" else "Memory Pool"
    
    print(f"[{mode}] Warming up...")
    for _ in range(10):
        _ = ap.zeros(shape, dtype=np.float32)
        
    print(f"[{mode}] Running {iterations} iterations...")
    
    start = time.perf_counter()
    for _ in range(iterations):
        a = ap.zeros(shape, dtype=np.float32)
        # a 销毁，触发 free
    end = time.perf_counter()
    
    avg_us = ((end - start) / iterations) * 1e6
    ops = iterations / (end - start)
    
    print(f"Result: {ops:.2f} ops/s | Latency: {avg_us:.2f} us")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Error: {e}")