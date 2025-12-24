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

import logging
import numpy as np
import asnumpy as ap

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_mean_1d():
    """测试一维数组的mean函数"""
    logger.info("Testing mean function - 1D arrays:")
    logger.info("=" * 50)
    
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
        
        logger.info(f"Test {i+1}:")
        logger.info(f"  Input: {np_arr}")
        logger.info(f"  NumPy result: {np_result}")
        logger.info(f"  AP result: {ap_result}")
        logger.info(f"  Match: {np.allclose(np_result, ap_result)}")
        logger.info("")


def test_mean_2d_axis0():
    """测试二维数组沿axis=0的mean函数"""
    logger.info("Testing mean function - 2D arrays (axis=0):")
    logger.info("=" * 50)
    
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
        
        logger.info(f"Test {i+1}:")
        logger.info(f"  Input shape: {np_arr.shape}")
        logger.info(f"  NumPy result: {np_result}")
        logger.info(f"  AP result: {ap_result_np}")
        logger.info(f"  Match: {np.allclose(np_result, ap_result_np)}")
        logger.info("")


def test_mean_2d_axis1():
    """测试二维数组沿axis=1的mean函数"""
    logger.info("Testing mean function - 2D arrays (axis=1):")
    logger.info("=" * 50)
    
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
        
        logger.info(f"Test {i+1}:")
        logger.info(f"  Input shape: {np_arr.shape}")
        logger.info(f"  NumPy result: {np_result}")
        logger.info(f"  AP result: {ap_result_np}")
        logger.info(f"  Match: {np.allclose(np_result, ap_result_np)}")
        logger.info("")


def test_mean_keepdims():
    """测试keepdims参数"""
    logger.info("Testing mean function - keepdims parameter:")
    logger.info("=" * 50)
    
    np_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    logger.info("Test 1: keepdims=True, axis=0")
    np_result_keep = np.mean(np_arr, axis=0, keepdims=True)
    ap_result_keep = ap.mean(ap_arr, axis=0, keepdims=True)
    ap_result_keep_np = ap_result_keep.to_numpy()
    
    logger.info(f"  Input shape: {np_arr.shape}")
    logger.info(f"  NumPy result shape: {np_result_keep.shape}")
    logger.info(f"  AP result shape: {ap_result_keep.shape}")
    logger.info(f"  NumPy result: {np_result_keep}")
    logger.info(f"  AP result: {ap_result_keep_np}")
    logger.info(f"  Shape match: {list(ap_result_keep.shape) == list(np_result_keep.shape)}")
    logger.info(f"  Value match: {np.allclose(np_result_keep, ap_result_keep_np)}")
    logger.info("")
    
    logger.info("Test 2: keepdims=False, axis=0")
    np_result_no_keep = np.mean(np_arr, axis=0, keepdims=False)
    ap_result_no_keep = ap.mean(ap_arr, axis=0, keepdims=False)
    ap_result_no_keep_np = ap_result_no_keep.to_numpy()
    
    logger.info(f"  NumPy result shape: {np_result_no_keep.shape}")
    logger.info(f"  AP result shape: {ap_result_no_keep.shape}")
    logger.info(f"  Shape match: {list(ap_result_no_keep.shape) == list(np_result_no_keep.shape)}")
    logger.info(f"  Value match: {np.allclose(np_result_no_keep, ap_result_no_keep_np)}")
    logger.info("")


def test_mean_3d():
    """测试三维数组的mean函数"""
    logger.info("Testing mean function - 3D arrays:")
    logger.info("=" * 50)
    
    np_arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    ap_arr = ap.ndarray.from_numpy(np_arr)
    
    # Test different axes
    for axis in [0, 1, 2, -1]:
        logger.info(f"Test axis={axis}:")
        np_result = np.mean(np_arr, axis=axis)
        ap_result = ap.mean(ap_arr, axis=axis, keepdims=False)
        ap_result_np = ap_result.to_numpy()
        
        logger.info(f"  Input shape: {np_arr.shape}")
        logger.info(f"  Result shape: {ap_result.shape}")
        logger.info(f"  Match: {np.allclose(np_result, ap_result_np)}")
        logger.info("")


def test_mean_int32():
    """测试int32数据类型"""
    logger.info("Testing mean function - int32 dtype:")
    logger.info("=" * 50)
    
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
        
        logger.info(f"Test {i+1}:")
        logger.info(f"  Input: {np_arr}")
        logger.info(f"  Input dtype: {np_arr.dtype}")
        logger.info(f"  AP result: {ap_result}")
        logger.info(f"  NumPy result: {np.mean(np_arr)}")
        logger.info("")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("AsNumpy Mean Function Test Suite")
    logger.info("=" * 60)
    logger.info("")
    
    test_mean_1d()
    test_mean_2d_axis0()
    test_mean_2d_axis1()
    test_mean_keepdims()
    test_mean_3d()
    test_mean_int32()
    
    logger.info("=" * 60)
    logger.info("All mean tests completed!")
    logger.info("=" * 60)

