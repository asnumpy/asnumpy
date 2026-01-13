import asnumpy as ap
import numpy as np

def test_add_workspace():
    print("TEST START: Add Operator with Workspace Verification")

    # 1. 准备数据
    M, N = 2048, 2048
    print(f"Creating Input A ({M}, {N}) and Input B ({N},) using Zeros...")
    a = ap.zeros((M, N), dtype=np.float32)
    b = ap.zeros((N,), dtype=np.float32)

    print("\n--- Execute Add (Watch logs below!) ---")
    
    # 2. 执行加法
    # 这次 C++ 里强行申请了 1024 字节，你一定能看到日志！
    c = ap.add(a, b)
    
    print("--- Execution Done ---")
    print(f"Output C ptr: {c.device_ptr}")

if __name__ == "__main__":
    test_add_workspace()