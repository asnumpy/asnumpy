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

def test_absolute():
    print("Testing absolute function:")
    print("=" * 50)
    
    test_cases = [
        np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
        np.array([-1.5, 2.7, -0.3, 4.0], dtype=np.float32),
        # np.array([], dtype=np.float32),
        np.array([-10], dtype=np.float32),
        np.array([[1, -2], [-3, 4]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = np.absolute(np_arr)
        ap_result = ap.absolute(ap_arr)
        
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input: {np_arr}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_sign():
    print("Testing sign function:")
    print("=" * 50)
    
    test_cases = [
        np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
        np.array([-1.5, 2.7, -0.3, 0.0, 4.0], dtype=np.float32),
        # np.array([], dtype=np.float32),
        np.array([-10, 0, 10], dtype=np.float32),
        np.array([[-1, 0], [0, 1]], dtype=np.float32),
    ]
    
    for i, arr in enumerate(test_cases):
        np_arr = arr
        ap_arr = ap.ndarray.from_numpy(np_arr)
        
        np_result = np.sign(np_arr)
        ap_result = ap.sign(ap_arr)
        
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  Input: {np_arr}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_heaviside():
    print("Testing heaviside function:")
    print("=" * 50)
    
    test_cases = [
        (np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32), 
         np.array([0.5], dtype=np.float32)),
        (np.array([-1.5, 0.0, 2.7], dtype=np.float32), 
         np.array([1.0], dtype=np.float32)),
        (np.array([0, 0, 0], dtype=np.float32), 
         np.array([0.5], dtype=np.float32)),
        (np.array([1, 2, 3], dtype=np.float32), 
         np.array([0.0], dtype=np.float32)),
        (np.array([-1, -2, -3], dtype=np.float32), 
         np.array([1.0], dtype=np.float32)),
        (np.array([[0, 1], [-1, 0]], dtype=np.float32), 
         np.array([[0.5, 0.5]], dtype=np.float32)),
    ]
    
    for i, (x1_np, x2_np) in enumerate(test_cases):
        x1_ap = ap.ndarray.from_numpy(x1_np)
        x2_ap = ap.ndarray.from_numpy(x2_np)
        
        np_result = np.heaviside(x1_np, x2_np)
        ap_result = ap.heaviside(x1_ap, x2_ap)
        
        ap_result_np = ap_result.to_numpy()
        
        print(f"Test {i+1}:")
        print(f"  x1: {x1_np}")
        print(f"  x2: {x2_np}")
        print(f"  NumPy result: {np_result}")
        print(f"  AP result: {ap_result_np}")
        print(f"  Match: {np.allclose(np_result, ap_result_np)}")
        print()

def test_broadcasting_heaviside():
    print("Testing heaviside broadcasting:")
    print("=" * 50)
    
    test_cases = [
        (np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float32), 
         np.array([0.5, 1.0, 0.0], dtype=np.float32)),
        (np.array([1, 2, 3], dtype=np.float32), 
         np.array([[0.5], [1.0]], dtype=np.float32)),
        (np.array([0], dtype=np.float32), 
         np.array([0.1, 0.2, 0.3], dtype=np.float32)),
    ]
    
    for i, (x1_np, x2_np) in enumerate(test_cases):
        x1_ap = ap.ndarray.from_numpy(x1_np)
        x2_ap = ap.ndarray.from_numpy(x2_np)
        
        try:
            np_result = np.heaviside(x1_np, x2_np)
            ap_result = ap.heaviside(x1_ap, x2_ap)
            ap_result_np = ap_result.to_numpy()
            
            print(f"Test {i+1}:")
            print(f"  x1 shape: {x1_np.shape}, x2 shape: {x2_np.shape}")
            print(f"  Result shape: {np_result.shape}")
            print(f"  Match: {np.allclose(np_result, ap_result_np)}")
            
        except Exception as e:
            print(f"Test {i+1}:")
            print(f"  x1 shape: {x1_np.shape}, x2 shape: {x2_np.shape}")
            print(f"  Error: {e}")
        print()

def test_dtype_compatibility():
    print("Testing dtype compatibility:")
    print("=" * 50)
    
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    
    for dtype in dtypes:
        try:
            np_arr = np.array([-2, -1, 0, 1, 2], dtype=dtype)
            ap_arr = ap.ndarray.from_numpy(np_arr)
            
            np_abs = np.absolute(np_arr)
            ap_abs = ap.absolute(ap_arr)
            
            np_sign = np.sign(np_arr)
            ap_sign = ap.sign(ap_arr)
            
            abs_match = np.allclose(np_abs, ap_abs.to_numpy())
            sign_match = np.allclose(np_sign, ap_sign.to_numpy())
            
            print(f"dtype {dtype.__name__}: absolute={abs_match}, sign={sign_match}")
            
        except Exception as e:
            print(f"dtype {dtype.__name__}: Error - {e}")

def main():
    print("Testing APU Math Functions")
    print("=" * 60)
    print()
    
    test_absolute()
    test_sign()
    test_heaviside()
    test_broadcasting_heaviside()
    test_dtype_compatibility()

if __name__ == "__main__":
    main()