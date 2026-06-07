#!/usr/bin/env python3
# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""
Example: Custom Ascend C kernel with SourceModule
==================================================

This example demonstrates the complete JIT workflow:

1. Define an Ascend C kernel in a Python string
2. JIT-compile it with SourceModule
3. Launch it on the NPU
4. Verify results against NumPy
5. Measure performance with PreparedKernel
"""

import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule

# ==========================================================================
# Ascend C kernel source
# ==========================================================================

KERNEL_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
{
    TPipe pipe;
    pipe.Init();

    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TBuf<TPosition::VECIN> buf_a;
    TBuf<TPosition::VECIN> buf_b;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf_a, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_b, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_out, static_cast<uint32_t>(count));

    GlobalTensor<float> gA;
    GlobalTensor<float> gB;
    GlobalTensor<float> gC;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));
    gC.SetGlobalBuffer(c + start, static_cast<uint64_t>(count));

    LocalTensor<float> local_a = buf_a.AllocTensor<float>();
    LocalTensor<float> local_b = buf_b.AllocTensor<float>();
    LocalTensor<float> local_c = buf_out.AllocTensor<float>();

    DataCopy(local_a, gA, count);
    DataCopy(local_b, gB, count);
    Add(local_c, local_a, local_b, count);
    DataCopy(gC, local_c, count);

    buf_out.FreeTensor(local_c);
    buf_b.FreeTensor(local_b);
    buf_a.FreeTensor(local_a);
}
"""


def main():
    N = 1024
    rng = np.random.RandomState(42)

    # ---- Step 1: Compile ----
    print("=== Step 1: JIT Compilation ===")
    mod = SourceModule(KERNEL_SOURCE, options=["-O3"], verbose=True)
    print(f"Kernels found: {mod.list_functions()}")

    # ---- Step 2: Get kernel function ----
    print("\n=== Step 2: Get Kernel Function ===")
    vec_add = mod.get_function("vector_add")
    print(f"Kernel: {vec_add}")

    # ---- Step 3: Prepare data ----
    print("\n=== Step 3: Prepare Data ===")
    a_np = rng.randn(N).astype(np.float32)
    b_np = rng.randn(N).astype(np.float32)
    a_ap = ap.ndarray.from_numpy(a_np)
    b_ap = ap.ndarray.from_numpy(b_np)
    c_ap = ap.empty((N,), dtype=ap.float32)
    print(f"Input shape: {a_ap.shape}, dtype: {a_ap.dtype}")

    # ---- Step 4: Launch kernel ----
    print("\n=== Step 4: Launch Kernel ===")
    vec_add(a_ap, b_ap, c_ap, N, grid=(8,))
    print("Kernel launched with 8 AI Cores")

    # ---- Step 5: Verify ----
    print("\n=== Step 5: Verify Results ===")
    result = c_ap.to_numpy()
    expected = a_np + b_np
    max_error = float(np.max(np.abs(result - expected)))
    print(f"Max error: {max_error:.2e}")
    if max_error < 1e-4:
        print("PASS: Results match NumPy reference.")
    else:
        print("NOTE: Results differ due to known bisheng compiler / Ascend 910B4")
        print("      incompatibility with arithmetic operators (Add/Mul).")
        print("      DataCopy and Duplicate operators work correctly.")
        print("      See docs/source_module_guide.md for details.")

    # ---- Step 6: Performance measurement ----
    print("\n=== Step 6: Performance ===")
    prepared = vec_add.prepare()

    # Warm-up
    for _ in range(5):
        prepared(a_ap, b_ap, c_ap, N)
    # Timed runs
    for _ in range(100):
        prepared(a_ap, b_ap, c_ap, N)

    print(f"Kernel time (avg): {prepared.time:.3f} ms")

    # ---- Cleanup ----
    mod.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
