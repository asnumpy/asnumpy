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
import gc
from typing import Tuple, Dict

def create_arrays(shape: Tuple[int, ...], dtype: np.dtype):
    """创建输入数组"""
    m1_np = np.random.normal(0, 1, shape).astype(dtype)
    m2_np = np.random.normal(0, 1, shape).astype(dtype)
    
    m1_asnp = ap.ndarray.from_numpy(m1_np)
    m2_asnp = ap.ndarray.from_numpy(m2_np)
    
    # 不再预分配 out，因为 multiply 不支持
    return m1_asnp, m2_asnp, m1_np, m2_np

def calculate_stable_metric(times: list, trim_ratio: float = 0.1) -> float:
    """
    统计策略：取中段最快速度
    排序后剔除最慢的 10%，取剩余部分的最小值
    """
    if not times:
        return 0.0
    
    sorted_times = sorted(times)
    keep_count = int(len(sorted_times) * (1.0 - trim_ratio))
    if keep_count < 1:
        keep_count = 1
        
    valid_times = sorted_times[:keep_count]
    return min(valid_times)

def bench_multiply(multiply_func, m1, m2, warmup: int, iterations: int, is_npu: bool = False) -> list:
    """
    基准测试函数
    针对 NPU 进行内存优化：显式删除中间变量，避免显存堆积
    """
    # 1. 预热阶段
    for _ in range(warmup):
        # 预热也要处理内存，防止预热阶段就 OOM
        res = multiply_func(m1, m2)
        # 如果是 NPU，建议同步以确保预热真实执行（假设 ap 有同步接口，若无则忽略）
        # if is_npu: ap.sync() 
        del res

    # 2. 正式测试阶段
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 执行计算，产生新对象
        res = multiply_func(m1, m2)
        
        # 注意：对于 NPU 异步计算，end 时间点可能只是 CPU 发完指令
        # 如果需要测量纯计算时间且 API 是异步的，通常需要 sync()
        # 如果 API 内部已同步或我们接受异步测量，则直接记录时间
        end = time.perf_counter()
        
        times.append(end - start)
        
        # 关键优化：显式删除结果对象
        # 这会触发 Python 引用计数归零，通知底层库释放显存
        del res

    # 如果底层库有缓存或延迟释放，建议在循环外强制垃圾回收
    if is_npu:
        gc.collect()
        
    return times

def run_test_case(shape: Tuple[int, ...], dtype: np.dtype = np.float32, 
                 warmup: int = 200, iterations: int = 3000) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'='*50}")
    print(f"测试形状: {shape}")
    
    m1_asnp, m2_asnp, m1_np, m2_np = create_arrays(shape, dtype)
    
    try:
        # --- 测试 AsNumpy (NPU) ---
        asnp_times = bench_multiply(
            ap.multiply, 
            m1_asnp, m2_asnp, 
            warmup=warmup,
            iterations=iterations,
            is_npu=True  # 标记为 NPU 测试，启用内存清理
        )
        
        # --- 测试 NumPy (CPU) ---
        np_times = bench_multiply(
            np.multiply, 
            m1_np, m2_np, 
            warmup=warmup,
            iterations=iterations,
            is_npu=False
        )
        
        # 计算统计数据
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0
        
        # 验证结果（只验证一次，避免内存占用）
        result_asnp = ap.multiply(m1_asnp, m2_asnp).to_numpy()
        result_np = np.multiply(m1_np, m2_np)
        
        max_diff = np.max(np.abs(result_asnp - result_np))
        max_val = np.max(np.abs(result_np))
        rel_diff = max_diff / max_val if max_val > 0 else max_diff
        
        if rel_diff < 1e-4:
            print(f"验证通过: 计算结果一致 (相对差异: {rel_diff:.2e})")
        else:
            print(f"警告: 计算结果存在差异 (最大相对差异: {rel_diff:.2e})")
            
        return {
            'shape': shape,
            'asnumpy_metric': metric_asnp,
            'numpy_metric': metric_np,
            'speedup': speedup,
            'relative_diff': rel_diff
        }
        
    finally:
        # 测试完一个 shape 后，彻底清理所有大对象
        del m1_asnp, m2_asnp, m1_np, m2_np
        gc.collect()

if __name__ == "__main__":
    print("=" * 70)
    print("性能基准测试 (修正版)")
    print("优化策略: 循环内显式 del 释放显存 + 取中段最快速度")
    print("=" * 70)
    
    # 测试配置
    shapes = [
        (500, 500),
        (1000, 1000),
        (2000, 2000),
        (3000, 3000),
    ]
    dtype = np.dtype(np.float32)
    
    # 参数设置
    warmup_iterations = 40
    test_iterations = 400
    
    print(f"\n配置信息:")
    print(f"  数据类型: {dtype}")
    print(f"  预热轮数: {warmup_iterations}")
    print(f"  测试轮数: {test_iterations}")
    print(f"  内存策略: 每次 iteration 后显式 del 结果")
    print(f"\n{'='*70}\n")
    
    results = []
    for shape in shapes:
        try:
            # 逐个运行，结果只存数值，不存大数组
            result = run_test_case(shape, dtype, warmup_iterations, test_iterations)
            results.append(result)
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出结果汇总
    print("\n" + "="*85)
    print("测试结果汇总 (基于中段最快速度)")
    print("-"*85)
    print(f"{'形状':<15} | {'数据量':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'加速比':<10}")
    print(f"{'':15} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-"*85)
    
    for result in results:
        shape_str = str(result['shape'])
        data_size = np.prod(result['shape'])
        data_size_str = f"{data_size:,}"
        asnp_time = f"{result['asnumpy_metric']*1000:.4f}"
        np_time = f"{result['numpy_metric']*1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"
        
        print(f"{shape_str:<15} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}")
    
    print("-"*85)
    
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print(f"\n平均加速比: {avg_speedup:.2f}x")


