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

import asnumpy as ap
import numpy as np
from loguru import logger

from utils import calculate_stable_metric


def create_arrays(shape: tuple[int, ...], dtype: np.dtype):
    """Create asnumpy and numpy test arrays"""
    # NumPy baseline data
    if dtype == np.bool_:
        # For bool type, generate random boolean values (~50% True, ~50% False)
        m1_np = np.random.rand(*shape) > 0.5
    else:
        # For other types, generate random data
        m1_np = np.random.rand(*shape).astype(dtype)

<<<<<<< HEAD
    # AsNumpy test data - converted from NumPy
=======
    # asnumpy测试数据 - 从 numpy 转换
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
    m1_asnp = ap.ndarray.from_numpy(m1_np)

    return m1_asnp, m1_np


def bench_all(all_func, m1, warmup: int, iterations: int, is_npu: bool = False) -> list:
    """
    Benchmark function for all reduction operation.
    Memory-optimized: explicitly delete intermediate variables to avoid OOM.
    """
    # 1. Warmup phase
    for _ in range(warmup):
        res = all_func(m1)
        del res  # Release immediately

    # 2. Benchmark phase
    times = []
    for _ in range(iterations):
        start = time.perf_counter()

<<<<<<< HEAD
        # Execute computation
=======
        # 执行计算
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        res = all_func(m1)
        end = time.perf_counter()
        times.append(end - start)

<<<<<<< HEAD
        # Explicitly delete result objects (all returns scalar, but keep consistent)
=======
        # 显式删除结果对象（虽然 all 返回标量，但保持一致性）
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        del res

    # Force garbage collection
    if is_npu:
        gc.collect()

    return times


def run_test_case(
<<<<<<< HEAD
    shape: tuple[int, ...],
    dtype: np.dtype,
    warmup: int = 40,
    iterations: int = 400,
) -> dict[str, float]:
    """Run a single test case"""
    logger.info(f"{'=' * 50}")
    logger.info(f"Test shape: {shape}, dtype: {dtype}")
=======
    shape: Tuple[int, ...], dtype: np.dtype, warmup: int = 40, iterations: int = 400
) -> Dict[str, float]:
    """运行单个测试用例"""
    print(f"{'=' * 50}")
    print(f"测试形状: {shape}, 类型: {dtype}")
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)

    m1_asnp, m1_np = create_arrays(shape, dtype)

    try:
<<<<<<< HEAD
        # --- Benchmark AsNumpy ---
        asnp_times = bench_all(
            ap.all,
            m1_asnp,
            warmup=warmup,
            iterations=iterations,
            is_npu=True
        )

        # --- Benchmark NumPy ---
        np_times = bench_all(
            np.all,
            m1_np,
            warmup=warmup,
            iterations=iterations,
            is_npu=False
        )

        # Calculate statistics
=======
        # --- 测试 AsNumpy ---
        asnp_times = bench_all(ap.all, m1_asnp, warmup=warmup, iterations=iterations, is_npu=True)

        # --- 测试 NumPy ---
        np_times = bench_all(np.all, m1_np, warmup=warmup, iterations=iterations, is_npu=False)

        # 计算统计信息
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        metric_asnp = calculate_stable_metric(asnp_times)
        metric_np = calculate_stable_metric(np_times)
        speedup = metric_np / metric_asnp if metric_asnp > 0 else 0

<<<<<<< HEAD
        # Verify result consistency
=======
        # 验证结果一致性
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        result_asnp = ap.all(m1_asnp)
        # all() returns a scalar, may need conversion
        if hasattr(result_asnp, "to_numpy"):
            result_asnp = result_asnp.to_numpy()
        result_np = np.all(m1_np)

<<<<<<< HEAD
        # For bool results, compare directly
=======
        # 对于 bool 结果，直接比较
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        if result_asnp == result_np:
            logger.info(f"Verification passed: results are consistent (result: {result_np})")
        else:
<<<<<<< HEAD
            logger.warning(f"Results differ (NumPy: {result_np}, AsNumpy: {result_asnp})")
=======
            print(f"警告: 计算结果存在差异 (NumPy: {result_np}, AsNumpy: {result_asnp})")
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)

        return {
            "shape": shape,
            "dtype": str(dtype),
            "asnumpy_metric": metric_asnp,
            "numpy_metric": metric_np,
            "speedup": speedup,
        }

    finally:
        # Explicitly clean up large objects for current shape
        del m1_asnp, m1_np
        gc.collect()


if __name__ == "__main__":
<<<<<<< HEAD
    logger.info("=" * 70)
    logger.info("README example code performance benchmark")
    logger.info("Test operation: all (reduction - check if all elements are True)")
    logger.info("Statistics strategy: after warmup, take mid-segment fastest speed (exclude slowest 10%)")
    logger.info("=" * 70)

    # Test configuration - multiple shapes and data types
    test_configs: list[tuple[tuple[int, ...], np.dtype]] = [
        # bool type tests
=======
    print("=" * 70)
    print("README 示例代码性能基准测试")
    print("测试操作: all (归约操作 - 检查所有元素是否为True)")
    print("统计策略: 预热后，取中段最快速度 (剔除最慢10%)")
    print("=" * 70)

    # 测试配置 - 多种形状和数据类型
    test_configs: List[Tuple[Tuple[int, ...], np.dtype]] = [
        # bool 类型测试
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        ((1000, 1000), np.bool_),
        ((2000, 2000), np.bool_),
        ((3000, 3000), np.bool_),
        # float32 type tests
        ((1000, 1000), np.float32),
        ((2000, 2000), np.float32),
        ((3000, 3000), np.float32),
    ]

<<<<<<< HEAD
    # Adjusted parameters: reduced iterations to fit NPU memory limits
    warmup_iterations = 40
    test_iterations = 400

    logger.info("\nConfiguration:")
    logger.info(f"  Warmup iterations: {warmup_iterations}")
    logger.info(f"  Test iterations: {test_iterations}")
    logger.info("  Statistics method: sort, exclude slowest 10%, take minimum")
    logger.info(f"\n{'=' * 70}\n")
=======
    # 按照建议调整参数：降低迭代次数以适配 NPU 显存限制
    warmup_iterations = 40
    test_iterations = 400

    print(f"\n配置信息:")
    print(f"  预热轮数: {warmup_iterations}")
    print(f"  测试轮数: {test_iterations}")
    print(f"  统计方法: 排序后剔除最慢10%，取最小值")
    print(f"\n{'=' * 70}\n")
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)

    results = []
    for shape, dtype in test_configs:
        try:
            result = run_test_case(shape, dtype, warmup_iterations, test_iterations)
            results.append(result)
        except Exception as e:
            logger.error(f"Test failed: {e}")
            import traceback

            traceback.print_exc()

<<<<<<< HEAD
    # Output summary results
=======
    # 输出汇总结果
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
    print("\n" + "=" * 95)
    print("Test results summary (based on mid-segment fastest speed)")
    print("-" * 95)
<<<<<<< HEAD
    print(f"{'Shape':<15} | {'Type':<10} | {'Data Size':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'Speedup':<10}")
=======
    print(
        f"{'形状':<15} | {'类型':<10} | {'数据量':<12} | {'AsNumpy':<12} | {'NumPy':<12} | {'加速比':<10}"
    )
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
    print(f"{'':15} | {'':10} | {'':12} | {'(ms)':<12} | {'(ms)':<12} | {'':10}")
    print("-" * 95)

    for result in results:
        shape_str = str(result["shape"])
        dtype_str = result["dtype"]
        data_size = np.prod(result["shape"])
        data_size_str = f"{data_size:,}"
        asnp_time = f"{result['asnumpy_metric'] * 1000:.4f}"
        np_time = f"{result['numpy_metric'] * 1000:.4f}"
        speedup_str = f"{result['speedup']:.2f}x"

        print(
<<<<<<< HEAD
            f"{shape_str:<15} | {dtype_str:<10} | {data_size_str:<12} "
            f"| {asnp_time:<12} | {np_time:<12} | {speedup_str}"
=======
            f"{shape_str:<15} | {dtype_str:<10} | {data_size_str:<12} | {asnp_time:<12} | {np_time:<12} | {speedup_str}"
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
        )

    print("-" * 95)

<<<<<<< HEAD
    # Statistics
    if results:
        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        max_speedup = max(r['speedup'] for r in results)

        logger.info(f"\n{'=' * 70}")
        logger.info("Performance statistics:")
        logger.info(f"  Average speedup: {avg_speedup:.2f}x")
        logger.info(f"  Maximum speedup: {max_speedup:.2f}x")
        logger.info(f"{'=' * 70}")

    logger.info("\nNotes:")
    logger.info("  • Testing both bool and float32 data types")
    logger.info("  • all() is a reduction operation that returns a scalar result")
    logger.info("  • Iterations adjusted to ensure memory safety")
    logger.info("  • Uses 'mid-segment fastest speed' algorithm to showcase NPU's true compute capability")
    logger.info("=" * 70)
=======
    # 统计信息
    if results:
        avg_speedup = sum(r["speedup"] for r in results) / len(results)
        max_speedup = max(r["speedup"] for r in results)

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
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files)
