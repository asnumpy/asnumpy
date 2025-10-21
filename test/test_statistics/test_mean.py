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

def test_mean_1d():
    """测试一维数组的mean函数"""
    print("Testing mean function - 1D arrays:")
    print("=" * 50)
    
    test_cases = [
        np.array([1, 2, 3, 4, 5], dtype=np.float32),
        np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        np.array([10, 20, 30], dtype=np.float32),
        np.array([-5, -3, -1, 1, 3, 5], dtype=np.float32),
        np.array([100], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = np.mean(np_arr)
        ap_result = ap.mean(ap_arr)
        
        print(f"Test {i+1}:")
        print(f"  Input: {np_arr}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result}")
        print(f"  Match: {np.allclose(np_result, ap_result)}")
        print()

def test_mean_2d_axis0():
    """测试二维数组沿axis=0的mean函数"""
    print("Testing mean function - 2D arrays (axis=0):")
    print("=" * 50)
    
    test_cases = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = np.mean(np_arr, axis=0)
        ap_result = ap.mean(ap_arr, axis=0, keepdims=False)
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input shape: {np_arr.shape}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_mean_2d_axis1():
    """测试二维数组沿axis=1的mean函数"""
    print("Testing mean function - 2D arrays (axis=1):")
    print("=" * 50)
    
    test_cases = [
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        np.array([[10, 20], [30, 40], [50, 60]], dtype=np.float32),
        np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = np.mean(np_arr, axis=1)
        ap_result = ap.mean(ap_arr, axis=1, keepdims=False)
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input shape: {np_arr.shape}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_mean_keepdims():
    """测试keepdims参数"""
    print("Testing mean function - keepdims parameter:")
    print("=" * 50)
    
    np_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    # Test keepdims=True
    print("Test 1: keepdims=True, axis=0")
    np_result_keep = np.mean(np_arr, axis=0, keepdims=True)
    ap_result_keep = ap.mean(ap_arr, axis=0, keepdims=True)
    ap_result_keep_np = ap_result_keep.to_numpy()
    
    print(f"  Input shape: {np_arr.shape}")
    print(f"  NumPy result shape: {np_result_keep.shape}")
    print(f"  AP result shape: {ap_result_keep.shape}")
    print(f"  NumPy result: {np_result_keep}")
    print(f"  AP result: {ap_result_keep_np}")
    print(f"  Shape match: {list(ap_result_keep.shape) == list(np_result_keep.shape)}")
    print(f"  Value match: {np.allclose(np_result_keep, ap_result_keep_np)}")
    print()
    
    # Test keepdims=False
    print("Test 2: keepdims=False, axis=0")
    np_result_no_keep = np.mean(np_arr, axis=0, keepdims=False)
    ap_result_no_keep = ap.mean(ap_arr, axis=0, keepdims=False)
    ap_result_no_keep_np = ap_result_no_keep.to_numpy()
    
    print(f"  NumPy result shape: {np_result_no_keep.shape}")
    print(f"  AP result shape: {ap_result_no_keep.shape}")
    print(f"  Shape match: {list(ap_result_no_keep.shape) == list(np_result_no_keep.shape)}")
    print(f"  Value match: {np.allclose(np_result_no_keep, ap_result_no_keep_np)}")
    print()

def test_mean_3d():
    """测试三维数组的mean函数"""
    print("Testing mean function - 3D arrays:")
    print("=" * 50)
    
    np_arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    # Test different axes
    for axis in [0, 1, 2, -1]:
        print(f"Test axis={axis}:")
        np_result = np.mean(np_arr, axis=axis)
        ap_result = ap.mean(ap_arr, axis=axis, keepdims=False)
        ap_result_np = ap_result.to_numpy()
        
        print(f"  Input shape: {np_arr.shape}")
        print(f"  Result shape: {ap_result.shape}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_mean_int32():
    """测试int32数据类型"""
    print("Testing mean function - int32 dtype:")
    print("=" * 50)
    
    test_cases = [
        np.array([1, 2, 3, 4, 5], dtype=np.int32),
        np.array([[1, 2], [3, 4]], dtype=np.int32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        # 注意：int32的mean可能与numpy不完全一致
        # 因为CANN算子的行为可能不同
        ap_result = ap.mean(ap_arr)
        
        print(f"Test {i+1}:")
        print(f"  Input: {np_arr}")
        print(f"  Input dtype: {np_arr.dtype}")
        print(f"  AP result: {ap_result}")
        print(f"  NumPy result: {np.mean(np_arr)}")
        print()

if __name__ == "__main__":
    print("=" * 60)
    print("AsNumpy Mean Function Test Suite")
    print("=" * 60)
    print()
    
    test_mean_1d()
    test_mean_2d_axis0()
    test_mean_2d_axis1()
    test_mean_keepdims()
    test_mean_3d()
    test_mean_int32()
    
    print("=" * 60)
    print("All mean tests completed!")
    print("=" * 60)

