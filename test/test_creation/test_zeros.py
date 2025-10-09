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
    print("Hello NPUArray zeros test\n")

    # Test zeros with different shapes and dtypes
    print("=== Test 1: Basic zeros with float32 ===")
    shape = (3, 4)
    dtype = np.float32
    zeros_npu = asnumpy.zeros(shape, dtype=dtype)
    print("NPUArray type:", type(zeros_npu))

    # Convert to numpy
    zeros_cpu = zeros_npu.to_numpy()
    print("Numpy type:", type(zeros_cpu))
    print("Numpy array:\n", zeros_cpu)
    print("Shape:", zeros_cpu.shape)
    print("Dtype:", zeros_cpu.dtype)

    # Compare with numpy's built-in zeros
    zeros_np = np.zeros(shape, dtype=dtype)
    print("Numpy zeros:\n", zeros_np)
    print("Allclose:", np.allclose(zeros_cpu, zeros_np))

    print("\n=== Test 2: Zeros with int32 ===")
    shape_int = (2, 3, 4)
    dtype_int = np.int32
    zeros_int_npu = asnumpy.zeros(shape_int, dtype=dtype_int)
    zeros_int_cpu = zeros_int_npu.to_numpy()
    print("Shape:", zeros_int_cpu.shape)
    print("Dtype:", zeros_int_cpu.dtype)
    print("Array:\n", zeros_int_cpu)

    # Compare with numpy
    zeros_int_np = np.zeros(shape_int, dtype=dtype_int)
    print("Allclose:", np.allclose(zeros_int_cpu, zeros_int_np))

    print("\n=== Test 3: Zeros with uint8 ===")
    shape_uint = (5, 5)
    dtype_uint = np.uint8
    zeros_uint_npu = asnumpy.zeros(shape_uint, dtype=dtype_uint)
    zeros_uint_cpu = zeros_uint_npu.to_numpy()
    print("Shape:", zeros_uint_cpu.shape)
    print("Dtype:", zeros_uint_cpu.dtype)
    print("Array:\n", zeros_uint_cpu)

    # Compare with numpy
    zeros_uint_np = np.zeros(shape_uint, dtype=dtype_uint)
    print("Allclose:", np.allclose(zeros_uint_cpu, zeros_uint_np))

    print("\n=== Test 4: Large array performance ===")
    large_shape = (100, 100)
    large_zeros_npu = asnumpy.zeros(large_shape, dtype=np.float32)
    large_zeros_cpu = large_zeros_npu.to_numpy()
    print("Large array shape:", large_zeros_cpu.shape)
    print("Large array sum:", np.sum(large_zeros_cpu))
    print("Expected sum: 0.0")

    print("\n=== Test 5: Different input formats ===")
    # Test with np.dtype wrapper
    zeros_dtype_npu = asnumpy.zeros((2, 2), dtype=np.dtype(np.float32))
    zeros_dtype_cpu = zeros_dtype_npu.to_numpy()
    print("With np.dtype wrapper:", zeros_dtype_cpu.dtype)

    # Test with string
    zeros_str_npu = asnumpy.zeros((2, 2), dtype='float32')
    zeros_str_cpu = zeros_str_npu.to_numpy()
    print("With string dtype:", zeros_str_cpu.dtype)

    print("\n=== Test 6: Edge cases ===")
    # Test 1D array
    zeros_1d_npu = asnumpy.zeros((10,), dtype=np.int32)
    zeros_1d_cpu = zeros_1d_npu.to_numpy()
    print("1D array shape:", zeros_1d_cpu.shape)
    print("1D array sum:", np.sum(zeros_1d_cpu))

    # Test 3D array
    zeros_3d_npu = asnumpy.zeros((2, 3, 4), dtype=np.float64)
    zeros_3d_cpu = zeros_3d_npu.to_numpy()
    print("3D array shape:", zeros_3d_cpu.shape)
    print("3D array sum:", np.sum(zeros_3d_cpu))

    print("\nAll zeros tests completed successfully!") 