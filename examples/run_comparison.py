import subprocess
import os
import sys
import re

def parse_result(output):
    """
    从子进程的输出中提取 ops/s 和 Latency。
    目标格式: "Result: 2534.85 ops/s | Latency: 394.50 us"
    """
    # 使用正则表达式提取数字
    ops_match = re.search(r"Result:\s+([\d\.]+)\s+ops/s", output)
    lat_match = re.search(r"Latency:\s+([\d\.]+)\s+us", output)
    
    ops = float(ops_match.group(1)) if ops_match else 0.0
    lat = float(lat_match.group(1)) if lat_match else 0.0
    
    return ops, lat

def run_test(mode_name, enable_pool):
    print(f"--- Testing {mode_name} ---")
    
    # 复制当前环境变量
    env = os.environ.copy()
    env["ASN_ENABLE_POOL"] = str(enable_pool)
    
    # 运行子进程，并捕获输出 (capture_output=True)
    try:
        result = subprocess.run(
            [sys.executable, "examples/bench_core.py"],
            env=env,
            capture_output=True,
            text=True
        )
        
        # 把原本的输出打印出来，让你能看到过程
        print(result.stdout)
        
        if result.stderr:
            print("Errors:", result.stderr)

        # 解析数值
        return parse_result(result.stdout)
        
    except Exception as e:
        print(f"Failed to run benchmark: {e}")
        return 0.0, 0.0

def main():
    print("========================================")
    print("🚀  AsNumpy Memory Pool Benchmark")
    print("========================================\n")

    # 1. 运行系统调用模式 (Baseline)
    ops_sys, lat_sys = run_test("System Call (Baseline)", "0")
    
    print("-" * 40 + "\n")

    # 2. 运行内存池模式 (Optimized)
    ops_pool, lat_pool = run_test("Memory Pool (Optimized)", "1")

    # 3. 计算加速比
    print("\n========================================")
    print("📊  Final Report")
    print("========================================")
    
    if ops_sys == 0 or ops_pool == 0:
        print("❌ Benchmark failed, cannot calculate speedup.")
        return

    speedup = ops_pool / ops_sys
    lat_reduction = (lat_sys - lat_pool) / lat_sys * 100
    lat_diff = lat_sys - lat_pool

    # 打印对比表格
    print(f"{'Metric':<20} | {'System Call':<15} | {'Memory Pool':<15} | {'Improvement':<15}")
    print("-" * 75)
    print(f"{'Throughput (ops/s)':<20} | {ops_sys:<15.2f} | {ops_pool:<15.2f} | {speedup:.2f}x 🚀")
    print(f"{'Latency (us)':<20} | {lat_sys:<15.2f} | {lat_pool:<15.2f} | -{lat_reduction:.1f}% ({lat_diff:.0f}us) ↓")
    print("========================================")

if __name__ == "__main__":
    main()