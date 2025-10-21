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

import numpy as np
import asnumpy as ap

def numpy_softmax(x, axis=-1):
    """NumPy实现的softmax（用于对比）"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # 数值稳定性
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def test_softmax_1d():
    """测试一维数组的softmax函数"""
    print("Testing softmax function - 1D arrays:")
    print("=" * 50)
    
    test_cases = [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        np.array([10.0, 20.0, 30.0], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = numpy_softmax(np_arr)
        ap_result = ap.softmax(ap_arr)
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input: {np_arr}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Sum of AP result: {np.sum(ap_result_np)} (should be ~1.0)")
        print(f"  Match: {np.allclose(np_result, ap_result_np, rtol=1e-5)}")
        print()

def test_softmax_2d_axis0():
    """测试二维数组沿axis=0的softmax函数"""
    print("Testing softmax function - 2D arrays (axis=0):")
    print("=" * 50)
    
    test_cases = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = numpy_softmax(np_arr, axis=0)
        ap_result = ap.softmax(ap_arr, axis=0)
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input shape: {np_arr.shape}")
        print(f"  NumPy result:\n{np_result}")
        print(f"  AP result:\n{ap_result_np}")
        print(f"  Column sums: {np.sum(ap_result_np, axis=0)} (should be ~1.0)")
        print(f"  Match: {np.allclose(np_result, ap_result_np, rtol=1e-5)}")
        print()

def test_softmax_2d_axis1():
    """测试二维数组沿axis=1的softmax函数"""
    print("Testing softmax function - 2D arrays (axis=1):")
    print("=" * 50)
    
    test_cases = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = numpy_softmax(np_arr, axis=1)
        ap_result = ap.softmax(ap_arr, axis=1)
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input shape: {np_arr.shape}")
        print(f"  NumPy result:\n{np_result}")
        print(f"  AP result:\n{ap_result_np}")
        print(f"  Row sums: {np.sum(ap_result_np, axis=1)} (should be ~1.0)")
        print(f"  Match: {np.allclose(np_result, ap_result_np, rtol=1e-5)}")
        print()

def test_softmax_negative_axis():
    """测试负数axis参数"""
    print("Testing softmax function - negative axis:")
    print("=" * 50)
    
    np_arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    # Test axis=-1 (should be same as axis=1 for 2D)
    print("Test axis=-1:")
    np_result = numpy_softmax(np_arr, axis=-1)
    ap_result = ap.softmax(ap_arr, axis=-1)
    ap_result_np = ap_result.to_numpy()
    
    print(f"  Input shape: {np_arr.shape}")
    print(f"  NumPy result:\n{np_result}")
    print(f"  AP result:\n{ap_result_np}")
    print(f"  Match: {np.allclose(np_result, ap_result_np, rtol=1e-5)}")
    print()

def test_softmax_3d():
    """测试三维数组的softmax函数"""
    print("Testing softmax function - 3D arrays:")
    print("=" * 50)
    
    np_arr = np.random.randn(2, 3, 4).astype(np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    # Test different axes
    for axis in [0, 1, 2, -1]:
        print(f"Test axis={axis}:")
        np_result = numpy_softmax(np_arr, axis=axis)
        ap_result = ap.softmax(ap_arr, axis=axis)
        ap_result_np = ap_result.to_numpy()
        
        print(f"  Input shape: {np_arr.shape}")
        print(f"  Result shape: {ap_result.shape}")
        print(f"  Match: {np.allclose(np_result, ap_result_np, rtol=1e-5)}")
        
        # Verify sum along axis
        sum_along_axis = np.sum(ap_result_np, axis=axis)
        all_close_to_one = np.allclose(sum_along_axis, 1.0, rtol=1e-5)
        print(f"  Sum along axis all ~1.0: {all_close_to_one}")
        print()

def test_softmax_large_values():
    """测试大数值的数值稳定性"""
    print("Testing softmax function - numerical stability:")
    print("=" * 50)
    
    # Large positive values
    print("Test 1: Large positive values")
    np_arr = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    np_result = numpy_softmax(np_arr)
    ap_result = ap.softmax(ap_arr)
    ap_result_np = ap_result.to_numpy()
    
    print(f"  Input: {np_arr}")
    print(f"  AP result: {ap_result_np}")
    print(f"  Sum: {np.sum(ap_result_np)}")
    print(f"  No NaN: {not np.any(np.isnan(ap_result_np))}")
    print(f"  No Inf: {not np.any(np.isinf(ap_result_np))}")
    print()
    
    # Mixed large and small values
    print("Test 2: Mixed values")
    np_arr = np.array([-100.0, 0.0, 100.0], dtype=np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    ap_result = ap.softmax(ap_arr)
    ap_result_np = ap_result.to_numpy()
    
    print(f"  Input: {np_arr}")
    print(f"  AP result: {ap_result_np}")
    print(f"  Sum: {np.sum(ap_result_np)}")
    print()

def test_softmax_batch():
    """测试批量处理（模拟深度学习场景）"""
    print("Testing softmax function - batch processing:")
    print("=" * 50)
    
    # Simulate a batch of 32 samples, each with 10 classes
    batch_size, num_classes = 32, 10
    np_logits = np.random.randn(batch_size, num_classes).astype(np.float32)
    ap_logits = ap.ndarray.from_numpy(np_logits)
    
    # Apply softmax along class dimension (axis=1)
    np_probs = numpy_softmax(np_logits, axis=1)
    ap_probs = ap.softmax(ap_logits, axis=1)
    ap_probs_np = ap_probs.to_numpy()
    
    print(f"  Batch size: {batch_size}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Input shape: {np_logits.shape}")
    print(f"  Output shape: {ap_probs.shape}")
    print(f"  All row sums ~1.0: {np.allclose(np.sum(ap_probs_np, axis=1), 1.0)}")
    print(f"  Match with NumPy: {np.allclose(np_probs, ap_probs_np, rtol=1e-5)}")
    print(f"  Max difference: {np.max(np.abs(np_probs - ap_probs_np))}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("AsNumpy Softmax Function Test Suite")
    print("=" * 60)
    print()
    
    test_softmax_1d()
    test_softmax_2d_axis0()
    test_softmax_2d_axis1()
    test_softmax_negative_axis()
    test_softmax_3d()
    test_softmax_large_values()
    test_softmax_batch()
    
    print("=" * 60)
    print("All softmax tests completed!")
    print("=" * 60)

