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

import asnumpy
import numpy as np
import time
from typing import Tuple, List, Dict

def create_arrays(shape: Tuple[int, ...], dtype: np.dtype):
    """创建asnumpy和numpy测试数组"""
    # asnumpy测试数据
    a_asnp = asnumpy.ones(shape=shape, dtype=dtype)
    b_asnp = asnumpy.ndarray.from_numpy(np.full(shape, 5, dtype=dtype))
    
    # numpy基准数据
    a_np = np.ones(shape, dtype=dtype)
    b_np = np.full(shape, 5, dtype=dtype)
    
    return a_asnp, b_asnp, a_np, b_np

def bench_add(func, *arrays, iterations: int = 10) -> List[float]:
    """基准测试函数"""
    times = []
    for _ in range(iterations):
        start = time.time()
        result = func(*arrays)
        times.append(time.time() - start)
    return times

def run_test_case(shape: Tuple[int, ...], dtype: np.dtype = np.int32, 
                 iterations: int = 10) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'='*50}")
    print(f"测试形状: {shape}")
    
    # 创建数组
    a_asnp, b_asnp, a_np, b_np = create_arrays(shape, dtype)
    
    # 执行性能测试
    asnp_times = bench_add(asnumpy.add, a_asnp, b_asnp, iterations=iterations)
    np_times = bench_add(np.add, a_np, b_np, iterations=iterations)
    
    # 计算统计信息
    avg_asnp = sum(asnp_times) / len(asnp_times)
    avg_np = sum(np_times) / len(np_times)
    speedup = avg_np / avg_asnp if avg_asnp > 0 else 0
    
    # 验证结果一致性
    assert np.array_equal(a_asnp.to_numpy(), a_np), "数组初始化不一致"
    assert np.array_equal(b_asnp.to_numpy(), b_np), "数组初始化不一致"
    assert np.array_equal(asnumpy.add(a_asnp, b_asnp).to_numpy(), 
                         np.add(a_np, b_np)), "计算结果不一致"
    
    print(f"验证通过: 形状 {shape} 的计算结果一致")
    
    return {
        'shape': shape,
        'asnumpy_avg': avg_asnp,
        'numpy_avg': avg_np,
        'speedup': speedup,
        'asnumpy_times': asnp_times,
        'numpy_times': np_times
    }

if __name__ == "__main__":
    print("Hello NPUArray - 性能基准测试")
    
    # 测试配置
    shapes = [
        (100,),               # 小规模测试
        (1000, 10),           # 中等规模
        (100, 100, 10),       # 三维测试
        (1000, 100, 10),      # 大规模测试
        (1000, 1000, 10),     # 更大规模测试
        (1000, 1000, 100)     # 超大规模测试
    ]
    dtype = np.dtype(np.int32)
    iterations = 1
    
    # 运行所有测试用例
    results = []
    for shape in shapes:
        result = run_test_case(shape, dtype, iterations)
        results.append(result)
    
    # 输出汇总结果
    print("\n" + "="*60)
    print("测试结果汇总 (单位: 秒)")
    print("-"*60)
    print(f"{'形状':<20} | {'asnumpy':<10} | {'numpy':<10} | 加速比")
    print("-"*60)
    
    for result in results:
        shape_str = str(result['shape'])
        asnp_time = f"{result['asnumpy_avg']:.4f}"
        np_time = f"{result['numpy_avg']:.4f}"
        speedup = f"{result['speedup']:.2f}x"
        print(f"{shape_str:<20} | {asnp_time:<10} | {np_time:<10} | {speedup}")
    
    print("-"*60)
    print("测试说明:")
    print(f"• 数据类型: {dtype}")
    print(f"• 迭代次数: {iterations} 次")
    print(f"• 广播功能: 已禁用（所有测试数组形状严格匹配）")
