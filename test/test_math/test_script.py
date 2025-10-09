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
from test_functions import UNARY_FUNCTIONS,BINARY_FUNCTIONS


def test_unary_functions():
    """通用单操作数函数测试"""
    for name, np_func, ap_func, test_cases in UNARY_FUNCTIONS:
        print(f"Testing {name} function:")
        print("=" * 50)
        
        passed = 0
        total = len(test_cases)
        
        for i, arr in enumerate(test_cases):
            np_arr = arr
            ap_arr = ap.ndarray.from_numpy(np_arr)
            
            np_result = np_func(np_arr)
            ap_result = ap_func(ap_arr)
            ap_result_np = ap_result.to_numpy()
            
            # 只在失败时输出详情
            if not np.allclose(np_result, ap_result_np, equal_nan=True):
                print(f"Test {i+1} FAILED:")
                print(f"  Input: {np_arr}")
                print(f"  NumPy result: {np_result}")
                print(f"  AP result: {ap_result_np}")
                print()
            else:
                passed += 1
                
        print(f"Passed: {passed}/{total} tests")
        print()

def test_binary_functions():
    """通用双操作数函数测试"""
    for name, np_func, ap_func, test_cases in BINARY_FUNCTIONS:
        print(f"Testing {name} function:")
        print("=" * 50)
        
        passed = 0
        total = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            # 处理单参数和双参数用例
            if isinstance(test_case, tuple) and len(test_case) == 2:
                x1_np, x2_np = test_case
            else:
                # 对于单参数函数错误注册的情况
                x1_np = test_case
                x2_np = np.zeros_like(x1_np)
                
            x1_ap = ap.ndarray.from_numpy(x1_np)
            x2_ap = ap.ndarray.from_numpy(x2_np)
            
            try:
                np_result = np_func(x1_np, x2_np)
                ap_result = ap_func(x1_ap, x2_ap)
                ap_result_np = ap_result.to_numpy()
                
                # 只在失败时输出详情
                if not np.allclose(np_result, ap_result_np, equal_nan=True):
                    print(f"Test {i+1} FAILED:")
                    print(f"  x1: {x1_np}")
                    print(f"  x2: {x2_np}")
                    print(f"  NumPy result: {np_result}")
                    print(f"  AP result: {ap_result_np}")
                    print()
                else:
                    passed += 1
                    
            except Exception as e:
                print(f"Test {i+1} ERROR:")
                print(f"  x1 shape: {x1_np.shape}, x2 shape: {x2_np.shape}")
                print(f"  Error: {e}")
                print()
                
        print(f"Passed: {passed}/{total} tests")
        print()

def test_dtype_compatibility():
    """数据类型兼容性测试"""
    print("Testing dtype compatibility:")
    print("=" * 50)
    
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    test_array = np.array([-2, -1, 0, 1, 2])
    
    total_tests = 0
    passed_tests = 0
    
    for dtype in dtypes:
        dtype_name = dtype.__name__
        # print(f"Testing dtype: {dtype_name}")
        # print("-" * 20)
        
        try:
            np_arr = test_array.astype(dtype)
            ap_arr = ap.ndarray.from_numpy(np_arr)
            
            # 测试所有单操作数函数
            for name, np_func, ap_func, _ in UNARY_FUNCTIONS:
                total_tests += 1
                try:
                    np_val = np_func(np_arr)
                    ap_val = ap_func(ap_arr).to_numpy()
                    
                    if np.allclose(np_val, ap_val, equal_nan=True):
                        passed_tests += 1
                    else:
                        print(f"  Function {name} FAILED:")
                        print(f"    Input: {np_arr}")
                        print(f"    NumPy result: {np_val}")
                        print(f"    AP result: {ap_val}")
                        print()
                except Exception as e:
                    print(f"  Function {name} ERROR:")
                    print(f"    Error: {e}")
                    print()
            
        except Exception as e:
            print(f"  Error in dtype conversion: {e}")
            # 当前数据类型的所有函数测试都失败
            total_tests += len(UNARY_FUNCTIONS)
            
        # print()
    
    # 打印总通过率
    print(f"Passed: {passed_tests}/{total_tests} tests")
    print()

def main():
    print("Testing NPU Math Functions")
    print("=" * 60)
    print()
    
    test_unary_functions()
    test_binary_functions()
    test_dtype_compatibility()
    
    print("=" * 60)
    print("All tests completed")

if __name__ == "__main__":
    main()