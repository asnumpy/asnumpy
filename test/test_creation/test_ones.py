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
    print("Hello NPUArray ones test\n")

    # Test ones with different shapes and dtypes
    print("=== Test 1: Basic ones with float32 ===")
    shape = (3, 4)
    dtype = np.float32
    ones_npu = asnumpy.ones(shape, dtype=dtype)
    print("NPUArray type:", type(ones_npu))

    # Convert to numpy
    ones_cpu = ones_npu.to_numpy()
    print("Numpy type:", type(ones_cpu))
    print("Numpy array:\n", ones_cpu)
    print("Shape:", ones_cpu.shape)
    print("Dtype:", ones_cpu.dtype)

    # Compare with numpy's built-in ones
    ones_np = np.ones(shape, dtype=dtype)
    print("Numpy ones:\n", ones_np)
    print("Allclose:", np.allclose(ones_cpu, ones_np))

    print("\n=== Test 2: Ones with int32 ===")
    shape_int = (2, 3, 4)
    dtype_int = np.int32
    ones_int_npu = asnumpy.ones(shape_int, dtype=dtype_int)
    ones_int_cpu = ones_int_npu.to_numpy()
    print("Shape:", ones_int_cpu.shape)
    print("Dtype:", ones_int_cpu.dtype)
    print("Array:\n", ones_int_cpu)

    # Compare with numpy
    ones_int_np = np.ones(shape_int, dtype=dtype_int)
    print("Allclose:", np.allclose(ones_int_cpu, ones_int_np))

    print("\n=== Test 3: Ones with uint8 ===")
    shape_uint = (5, 5)
    dtype_uint = np.uint8
    ones_uint_npu = asnumpy.ones(shape_uint, dtype=dtype_uint)
    ones_uint_cpu = ones_uint_npu.to_numpy()
    print("Shape:", ones_uint_cpu.shape)
    print("Dtype:", ones_uint_cpu.dtype)
    print("Array:\n", ones_uint_cpu)

    # Compare with numpy
    ones_uint_np = np.ones(shape_uint, dtype=dtype_uint)
    print("Allclose:", np.allclose(ones_uint_cpu, ones_uint_np))

    print("\n=== Test 4: Large array performance ===")
    large_shape = (100, 100)
    large_ones_npu = asnumpy.ones(large_shape, dtype=np.float32)
    large_ones_cpu = large_ones_npu.to_numpy()
    print("Large array shape:", large_ones_cpu.shape)
    print("Large array sum:", np.sum(large_ones_cpu))
    print("Expected sum:", large_shape[0] * large_shape[1])

    print("\n=== Test 5: Different input formats ===")
    # Test with np.dtype wrapper
    ones_dtype_npu = asnumpy.ones((2, 2), dtype=np.dtype(np.float32))
    ones_dtype_cpu = ones_dtype_npu.to_numpy()
    print("With np.dtype wrapper:", ones_dtype_cpu.dtype)

    # Test with string
    ones_str_npu = asnumpy.ones((2, 2), dtype='float32')
    ones_str_cpu = ones_str_npu.to_numpy()
    print("With string dtype:", ones_str_cpu.dtype)

    print("\nAll ones tests completed successfully!") 