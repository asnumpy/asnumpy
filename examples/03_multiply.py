# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import asnumpy as ap
import numpy as np
import time
from typing import Tuple, List, Dict

def create_arrays(shape: Tuple[int, ...], dtype: np.dtype):
    """创建asnumpy和numpy测试数组"""
    # numpy 基准数据
    m1_np = np.random.normal(0, 1, shape).astype(dtype)
    m2_np = np.random.normal(0, 1, shape).astype(dtype)
    
    # asnumpy测试数据 - 从 numpy 转换
    m1_asnp = ap.ndarray.from_numpy(m1_np)
    m2_asnp = ap.ndarray.from_numpy(m2_np)
    
    return m1_asnp, m2_asnp, m1_np, m2_np

def bench_multiply(multiply_func, m1, m2, iterations: int = 10) -> List[float]:
    """基准测试函数 - 只测试 multiply 操作"""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        elementwise_product = multiply_func(m1, m2)
        end = time.perf_counter()
        times.append(end - start)
    return times

def run_test_case(shape: Tuple[int, ...], dtype: np.dtype = np.float32, 
                 iterations: int = 20) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'='*50}")
    print(f"测试形状: {shape}")
    
    # 创建数组
    m1_asnp, m2_asnp, m1_np, m2_np = create_arrays(shape, dtype)
    
    # 执行性能测试 - AsNumpy
    asnp_times = bench_multiply(
        ap.multiply, 
        m1_asnp, m2_asnp, 
        iterations=iterations
    )
    
    # 执行性能测试 - NumPy
    np_times = bench_multiply(
        np.multiply, 
        m1_np, m2_np, 
        iterations=iterations
    )
    
    # 计算统计信息
    avg_asnp = sum(asnp_times) / len(asnp_times)
    avg_np = sum(np_times) / len(np_times)
    min_asnp = min(asnp_times)
    min_np = min(np_times)
    speedup = avg_np / avg_asnp if avg_asnp > 0 else 0
    
    # 验证结果一致性
    result_asnp = ap.multiply(m1_asnp, m2_asnp).to_numpy()
    result_np = np.multiply(m1_np, m2_np)
    
    # 计算最大绝对差异和相对差异
    max_diff = np.max(np.abs(result_asnp - result_np))
    max_val = np.max(np.abs(result_np))
    rel_diff = max_diff / max_val if max_val > 0 else max_diff
    
    if rel_diff < 1e-4:
        print(f"验证通过: 形状 {shape} 的计算结果一致 (相对差异: {rel_diff:.2e})")
    else:
        print(f"警告: 计算结果存在差异 (最大相对差异: {rel_diff:.2e})")
    
    return {
        'shape': shape,
        'asnumpy_avg': avg_asnp,
        'numpy_avg': avg_np,
        'asnumpy_min': min_asnp,
        'numpy_min': min_np,
        'speedup': speedup,
        'asnumpy_times': asnp_times,
        'numpy_times': np_times,
        'relative_diff': rel_diff
    }

if __name__ == "__main__":
    print("=" * 70)
    print("README 示例代码性能基准测试")
    print("测试操作: multiply (元素级乘法)")
    print("=" * 70)
    
    # 测试配置 - 使用二维张量获得最佳性能表现
    shapes = [
        (500, 500),            # 中等规模
        (1000, 1000),          # 大规模测试
        (2000, 2000),          # 更大规模测试
        (3000, 3000),          # 超大规模测试
    ]
    dtype = np.dtype(np.float32)  # 使用 float32 兼容 NPU
    iterations = 50  # 增加迭代次数以获得更稳定的结果
    
    print(f"\n配置信息:")
    print(f"  数据类型: {dtype}")
    print(f"  迭代次数: {iterations}")
    print(f"  测试操作: multiply(m1, m2)")
    print(f"\n{'='*70}\n")
    
    # 运行所有测试用例
    results = []
    for shape in shapes:
        try:
            result = run_test_case(shape, dtype, iterations)
            results.append(result)
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出汇总结果
    print("\n" + "="*85)
    print("测试结果汇总")
    print("-"*85)
    print(f"{'形状':<15} | {'数据量':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'加速比':<10}")
    print(f"{'':15} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-"*85)
    
    for result in results:
        shape_str = str(result['shape'])
        data_size = np.prod(result['shape'])
        data_size_str = f"{data_size:,}"
        asnp_time = f"{result['asnumpy_avg']*1000:.4f}"
        np_time = f"{result['numpy_avg']*1000:.4f}"
        speedup = result['speedup']
        
        if speedup >= 1.0:
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = f"{speedup:.2f}x"
        
        print(f"{shape_str:<15} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}")
    
    print("-"*85)
    
    # 统计信息
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)
        min_speedup = min(r['speedup'] for r in results)
        
        print(f"\n{'='*70}")
        print("性能统计:")
        print(f"  平均加速比: {avg_speedup:.2f}x")
        print(f"  最大加速比: {max_speedup:.2f}x (形状: {[r['shape'] for r in results if r['speedup'] == max_speedup][0]})")
        print(f"  最小加速比: {min_speedup:.2f}x (形状: {[r['shape'] for r in results if r['speedup'] == min_speedup][0]})")
        print(f"{'='*70}")
    
    print("\n注意事项:")
    print("  • 使用 float32 数据类型以兼容 NPU")
    print("  • 只测试 multiply 操作的性能")
    print("  • 数据已预先转换到 NPU（转换时间不计入）")
    print("  • 使用 time.perf_counter() 测量高精度时间")
    print("="*70)

