import asnumpy as ap
import time
import os

def run_workload(iterations=1000):
    """模拟一个高频创建和销毁不同大小数组的负载"""
    start_time = time.time()
    
    # 数组形状列表，模拟不同维度的需求
    shapes = [
        (1024, 1024),  # 4MB
        (512, 1024),   # 2MB
        (2048, 512),   # 4MB
        (256, 256),    # 256KB
        (1024, 256),   # 1MB
    ]
    
    for i in range(iterations):
        # 循环创建并立即释放数组
        target_shape = shapes[i % len(shapes)]
        temp = ap.zeros(target_shape, dtype='float32')
        # 执行一个简单的计算操作，确保显存被真实触碰
        # (可选：如果有运算 API 的话，比如 res = temp + 1)
        del temp
        
    end_time = time.time()
    return end_time - start_time

def benchmark():
    # 检测当前是否启用了内存池
    pool_enabled = os.environ.get("ASN_ENABLE_POOL", "1") != "0"
    mode_name = "【内存池模式】" if pool_enabled else "【系统直调 (aclrtMalloc) 模式】"
    
    print(f"\n🚀 开始执行 {mode_name} 测试...")
    
    num_iterations = 2000
    elapsed = run_workload(num_iterations)
    
    avg_time_ms = (elapsed / num_iterations) * 1000
    throughput = num_iterations / elapsed
    
    print("-" * 50)
    print(f"总计完成次数: {num_iterations}")
    print(f"总耗时:      {elapsed:.4f} 秒")
    print(f"平均单次耗时: {avg_time_ms:.4f} 毫秒")
    print(f"吞吐量:      {throughput:.2f} ops/sec")
    print("-" * 50)
    
    return avg_time_ms, throughput

if __name__ == "__main__":
    benchmark()