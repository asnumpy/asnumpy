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

import gc
import time
import numpy as np
import asnumpy as ap

from typing import Dict, List
from utils import calculate_stable_metric


def bench_linspace(linspace_func, start: float, end: float, steps: int,
                   dtype: np.dtype, warmup: int, iterations: int, 
                   is_npu: bool = False) -> list:
    """
    基准测试函数 - 针对 linspace 操作
    包含内存优化：显式删除中间变量避免 OOM
    """
    # 1. 预热阶段
    for _ in range(warmup):
        res = linspace_func(start, end, steps, dtype=dtype)
        del res  # 及时释放

    # 2. 正式测试阶段
    times = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        
        # 执行计算
        res = linspace_func(start, end, steps, dtype=dtype)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
        
        # 关键优化：显式删除结果对象，避免显存堆积
        del res

    # 强制垃圾回收
    if is_npu:
        gc.collect()
        
    return times


def run_test_case(steps: int, start: float = 0.0, end: float = 100.0,
                  dtype: np.dtype = np.float32, 
                  warmup: int = 40, iterations: int = 400) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'=' * 50}")
    print(f"测试步数: {steps:,} (范围: [{start}, {end}])")
    
    try:
        # --- 测试 AsNumpy ---
        asnp_times = bench_linspace(
            ap.linspace, 
            start, end, steps, dtype,
            warmup=warmup,
            iterations=iterations,
            is_npu=True
        )
        
        # --- 测试 NumPy ---
        np_times = bench_linspace(
            np.linspace, 
            start, end, steps, dtype,
            warmup=warmup,
            iterations=iterations,
            is_npu=False
        )
        
        # 计算统计信息
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0
        
        # 验证结果一致性
        result_asnp = ap.linspace(start, end, steps, dtype=dtype).to_numpy()
        result_np = np.linspace(start, end, steps, dtype=dtype)
        
        max_diff = np.max(np.abs(result_asnp - result_np))
        max_val = np.max(np.abs(result_np))
        rel_diff = max_diff / max_val if max_val > 0 else max_diff
        
        if rel_diff < 1e-4:
            print(f"验证通过: 计算结果一致 (相对差异: {rel_diff:.2e})")
        else:
            print(f"警告: 计算结果存在差异 (最大相对差异: {rel_diff:.2e})")
            
        return {
            'steps': steps,
            'asnumpy_metric': metric_asnp,
            'numpy_metric': metric_np,
            'speedup': speedup,
            'relative_diff': rel_diff
        }
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 强制垃圾回收
        gc.collect()


if __name__ == "__main__":
    print("=" * 70)
    print("README 示例代码性能基准测试")
    print("测试操作: linspace (创建等间距数列)")
    print("统计策略: 预热后，取中段最快速度 (剔除最慢10%)")
    print("=" * 70)
    
    # 测试配置 - 步数与其他示例的数据量对应
    test_steps = [
        250000,     # 对应 (500, 500) 数据量
        1000000,    # 对应 (1000, 1000) 数据量
        4000000,    # 对应 (2000, 2000) 数据量
        9000000,    # 对应 (3000, 3000) 数据量
    ]
    dtype = np.dtype(np.float32)
    start_val = 0.0
    end_val = 100.0
    
    # 按照建议调整参数：降低迭代次数以适配 NPU 显存限制
    warmup_iterations = 40
    test_iterations = 400
    
    print(f"\n配置信息:")
    print(f"  数据类型: {dtype}")
    print(f"  起始值: {start_val}")
    print(f"  终止值: {end_val}")
    print(f"  预热轮数: {warmup_iterations}")
    print(f"  测试轮数: {test_iterations}")
    print(f"  统计方法: 排序后剔除最慢10%，取最小值")
    print(f"\n{'=' * 70}\n")
    
    results = []
    for steps in test_steps:
        result = run_test_case(steps, start_val, end_val, dtype, 
                               warmup_iterations, test_iterations)
        if result is not None:
            results.append(result)
    
    # 输出汇总结果
    print("\n" + "=" * 85)
    print("测试结果汇总 (基于中段最快速度)")
    print("-" * 85)
    print(f"{'步数':<15} | {'数据量':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'加速比':<10}")
    print(f"{'':15} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 85)
    
    for result in results:
        steps_str = f"{result['steps']:,}"
        data_size_str = f"{result['steps']:,}"
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"
        
        print(f"{steps_str:<15} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}")
    
    print("-" * 85)
    
    # 统计信息
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)
        
        print(f"\n{'=' * 70}")
        print("性能统计:")
        print(f"  平均加速比: {avg_speedup:.2f}x")
        print(f"  最大加速比: {max_speedup:.2f}x")
        print(f"{'=' * 70}")
    
    print("\n注意事项:")
    print("  • 使用 float32 数据类型以兼容 NPU")
    print("  • linspace 生成指定范围内等间距的一维数组")
    print("  • 已调整迭代次数以保证显存安全")
    print("  • 采用'取中段最快速度'算法，展示 NPU 真实计算能力")
    print("=" * 70)
