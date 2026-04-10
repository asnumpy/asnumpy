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
from typing import Dict, List, Tuple

import numpy as np
import asnumpy as ap

# 导入公共工具函数
from utils import calculate_stable_metric


def create_arrays(shape: Tuple[int, ...], dtype: np.dtype):
    """创建asnumpy和numpy测试数组"""
    # numpy 基准数据
    if dtype == np.bool_:
        # 对于 bool 类型，生成随机布尔值（大约一半True一半False）
        m1_np = np.random.rand(*shape) > 0.5
    else:
        # 对于其他类型，生成随机数据
        m1_np = np.random.rand(*shape).astype(dtype)
    
    # asnumpy测试数据 - 从 numpy 转换
    m1_asnp = ap.ndarray.from_numpy(m1_np)
    
    return m1_asnp, m1_np


def bench_all(all_func, m1, warmup: int, iterations: int, is_npu: bool = False) -> list:
    """
    基准测试函数 - 针对 all 归约操作
    包含内存优化：显式删除中间变量避免 OOM
    """
    # 1. 预热阶段
    for _ in range(warmup):
        res = all_func(m1)
        del res  # 及时释放

    # 2. 正式测试阶段
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        
        # 执行计算
        res = all_func(m1) 
        end = time.perf_counter()
        times.append(end - start)
        
        # 显式删除结果对象（虽然 all 返回标量，但保持一致性）
        del res

    # 强制垃圾回收
    if is_npu:
        gc.collect()
        
    return times


def run_test_case(shape: Tuple[int, ...], dtype: np.dtype, 
                  warmup: int = 40, iterations: int = 400) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'=' * 50}")
    print(f"测试形状: {shape}, 类型: {dtype}")
    
    m1_asnp, m1_np = create_arrays(shape, dtype)
    
    try:
        # --- 测试 AsNumpy ---
        asnp_times = bench_all(
            ap.all, 
            m1_asnp, 
            warmup=warmup,
            iterations=iterations,
            is_npu=True
        )
        
        # --- 测试 NumPy ---
        np_times = bench_all(
            np.all, 
            m1_np, 
            warmup=warmup,
            iterations=iterations,
            is_npu=False
        )
        
        # 计算统计信息
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0
        
        # 验证结果一致性
        result_asnp = ap.all(m1_asnp)
        # all() 返回的是标量，可能需要转换
        if hasattr(result_asnp, "to_numpy"):
            result_asnp = result_asnp.to_numpy()
        result_np = np.all(m1_np)
        
        # 对于 bool 结果，直接比较
        if result_asnp == result_np:
            print(f"验证通过: 计算结果一致 (结果: {result_np})")
        else:
            print(f"警告: 计算结果存在差异 (NumPy: {result_np}, AsNumpy: {result_asnp})")
            
        return {
            'shape': shape,
            'dtype': str(dtype),
            'asnumpy_metric': metric_asnp,
            'numpy_metric': metric_np,
            'speedup': speedup,
        }
        
    finally:
        # 显式清理当前 shape 的大对象
        del m1_asnp, m1_np
        gc.collect()


if __name__ == "__main__":
    print("=" * 70)
    print("README 示例代码性能基准测试")
    print("测试操作: all (归约操作 - 检查所有元素是否为True)")
    print("统计策略: 预热后，取中段最快速度 (剔除最慢10%)")
    print("=" * 70)
    
    # 测试配置 - 多种形状和数据类型
    test_configs: List[Tuple[Tuple[int, ...], np.dtype]] = [
        # bool 类型测试
        ((1000, 1000), np.bool_),
        ((2000, 2000), np.bool_),
        ((3000, 3000), np.bool_),
        # float32 类型测试
        ((1000, 1000), np.float32),
        ((2000, 2000), np.float32),
        ((3000, 3000), np.float32),
    ]
    
    # 按照建议调整参数：降低迭代次数以适配 NPU 显存限制
    warmup_iterations = 40
    test_iterations = 400
    
    print(f"\n配置信息:")
    print(f"  预热轮数: {warmup_iterations}")
    print(f"  测试轮数: {test_iterations}")
    print(f"  统计方法: 排序后剔除最慢10%，取最小值")
    print(f"\n{'=' * 70}\n")
    
    results = []
    for shape, dtype in test_configs:
        try:
            result = run_test_case(shape, dtype, warmup_iterations, test_iterations)
            results.append(result)
        except Exception as e:
            print(f"测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 输出汇总结果
    print("\n" + "=" * 95)
    print("测试结果汇总 (基于中段最快速度)")
    print("-" * 95)
    print(f"{'形状':<15} | {'类型':<10} | {'数据量':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'加速比':<10}")
    print(f"{'':15} | {'':10} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 95)
    
    for result in results:
        shape_str = str(result['shape'])
        dtype_str = result['dtype']
        data_size = np.prod(result['shape'])
        data_size_str = f"{data_size:,}"
        # 修复：算术操作符两侧增加空格
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"
        
        print(f"{shape_str:<15} | {dtype_str:<10} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}")
    
    print("-" * 95)
    
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
    print("  • 测试 bool 和 float32 两种数据类型")
    print("  • all() 是归约操作，返回标量结果")
    print("  • 已调整迭代次数以保证显存安全")
    print("  • 采用'取中段最快速度'算法，展示 NPU 真实计算能力")
    print("=" * 70)
