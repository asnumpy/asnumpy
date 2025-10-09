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

if __name__ == "__main__":
    print("=== Arange Test ===\n")
    
    # Test 1: Basic integer range
    print("1. Basic integer range test:")
    # arr1 = asnumpy.arange(0, 10, 1, dtype=np.dtype(np.int32))
    arr1 = asnumpy.arange(0, 10, 1, dtype=np.int32)
    result1 = arr1.to_numpy()
    print(f"   arange(0, 10, 1, int32): {result1}")
    print(f"   Shape: {result1.shape}, Element count: {len(result1)}")
    
    # Test 2: Float step
    print("\n2. Float step test:")
    # arr2 = asnumpy.arange(0, 5, 0.5, dtype=np.dtype(np.float32))
    arr2 = asnumpy.arange(0, 5, 0.5, dtype=np.float32)
    result2 = arr2.to_numpy()
    print(f"   arange(0, 5, 0.5, float32): {result2}")
    print(f"   Shape: {result2.shape}, Element count: {len(result2)}")
    
    # Test 3: Negative step
    print("\n3. Negative step test:")
    # arr3 = asnumpy.arange(10, 0, -1, dtype=np.dtype(np.int64))
    arr3 = asnumpy.arange(10, 0, -1, dtype=np.int64)
    result3 = arr3.to_numpy()
    print(f"   arange(10, 0, -1, int64): {result3}")
    print(f"   Shape: {result3.shape}, Element count: {len(result3)}")
    
    # Test 4: Float range
    print("\n4. Float range test:")
    # arr4 = asnumpy.arange(5.5, -3.2, -1.1, dtype=np.dtype(np.float32))
    arr4 = asnumpy.arange(5.5, -3.2, -1.1, dtype=np.float32)
    result4 = arr4.to_numpy()
    print(f"   arange(5.5, -3.2, -1.1, float32): {result4}")
    print(f"   Shape: {result4.shape}, Element count: {len(result4)}")
    
    # Test 5: Empty array
    print("\n5. Empty array test:")
    try:
        # arr5 = asnumpy.arange(0, 0, 1, dtype=np.dtype(np.int32))
        arr5 = asnumpy.arange(0, 0, 1, dtype=np.int32)
        result5 = arr5.to_numpy()
        print(f"   arange(0, 0, 1, int32): {result5}")
        print(f"   Shape: {result5.shape}, Element count: {len(result5)}")
    except Exception as e:
        print(f"Empty array test exception: {e}")
    
    # Test 6: Consecutive calls test
    print("\n6. Consecutive calls test:")
    try:
        # a = asnumpy.arange(0, 3, 1, dtype=np.dtype(np.int32))
        a = asnumpy.arange(0, 3, 1, dtype=np.int32)
        # b = asnumpy.arange(0, 2, 0.5, dtype=np.dtype(np.float32))
        b = asnumpy.arange(0, 2, 0.5, dtype=np.float32)
        # c = asnumpy.arange(5, 0, -1, dtype=np.dtype(np.int64))
        c = asnumpy.arange(5, 0, -1, dtype=np.int64)
        
        print(f"   First call: {a.to_numpy()}")
        print(f"   Second call: {b.to_numpy()}")
        print(f"   Third call: {c.to_numpy()}")
        print("Consecutive calls successful")
    except Exception as e:
        print(f"Consecutive calls failed: {e}")
    
    # Test 7: Comparison with NumPy
    print("\n7. Comparison with NumPy test:")
    # asnumpy_result = asnumpy.arange(0, 5, 1, dtype=np.dtype(np.int32)).to_numpy()
    asnumpy_result = asnumpy.arange(0, 5, 1, dtype=np.int32).to_numpy() 
    numpy_result = np.arange(0, 5, dtype=np.int32)
    print(f"   asnumpy result: {asnumpy_result}")
    print(f"   numpy result: {numpy_result}")
    if np.array_equal(asnumpy_result, numpy_result):
        print("Results are consistent")
    else:
        print("Results are inconsistent")
    
    print("\n=== Tests Completed ===")
