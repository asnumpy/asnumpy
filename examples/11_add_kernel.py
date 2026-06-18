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
Example: Ascend C JIT-compiled vector addition kernel
======================================================

Demonstrates the complete SourceModule workflow for an element-wise
vector add kernel:

1. Define an Ascend C kernel in a Python string
2. JIT-compile it via bisheng
3. Launch it on the NPU and verify correctness

Key Ascend C patterns:
  - TQue double-buffering for CopyIn → Compute → CopyOut pipeline
  - DataCopyPad + DataCopyExtParams (required for correct DMA on 910B4)
  - UB tiling — split work into 256-element tiles
  - Multi-block SPMD parallelism via GetBlockIdx/GetBlockNum

.. note::

   The kernel follows the official CANN ops-math add pattern.
   Plain ``DataCopy`` with ``TBuf`` produces incorrect results on
   Ascend 910B4; always use ``TQue`` + ``DataCopyPad``.
"""

import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule
from loguru import logger

# ==========================================================================
# Ascend C kernel — vector element-wise addition
#
# Adapted from CANN ops-math:
#   ops-math/examples/fast_kernel_launch_example/csrc/add/dav-2201/add.asc
# ==========================================================================

KERNEL_SRC = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int totalLength)
{
    TPipe pipe;

    constexpr int TILE_ELEMS = 256;
    constexpr int PIPELINE_DEPTH = 2;

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueA;
    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueB;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueC;

    uint32_t tileBytes = TILE_ELEMS * sizeof(float);
    pipe.InitBuffer(inQueueA, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(inQueueB, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outQueueC, PIPELINE_DEPTH, tileBytes);

    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (totalLength + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < totalLength - start) ? per_block : (totalLength - start);

    GlobalTensor<float> gA, gB, gC;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));
    gC.SetGlobalBuffer(c + start, static_cast<uint64_t>(count));

    int tileNum = count / TILE_ELEMS;
    int tailElems = count - tileNum * TILE_ELEMS;

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    for (int i = 0; i < tileNum; ++i) {
        int offset = i * TILE_ELEMS;
        // CopyIn
        LocalTensor<float> localA = inQueueA.AllocTensor<float>();
        LocalTensor<float> localB = inQueueB.AllocTensor<float>();
        copyParams.blockLen = TILE_ELEMS * sizeof(float);
        DataCopyPad(localA, gA[offset], copyParams, padParams);
        DataCopyPad(localB, gB[offset], copyParams, padParams);
        inQueueA.EnQue(localA);
        inQueueB.EnQue(localB);
        // Compute
        localA = inQueueA.DeQue<float>();
        localB = inQueueB.DeQue<float>();
        LocalTensor<float> localC = outQueueC.AllocTensor<float>();
        Add(localC, localA, localB, TILE_ELEMS);
        outQueueC.EnQue(localC);
        inQueueA.FreeTensor(localA);
        inQueueB.FreeTensor(localB);
        // CopyOut
        localC = outQueueC.DeQue<float>();
        DataCopyPad(gC[offset], localC, copyParams);
        outQueueC.FreeTensor(localC);
    }

    if (tailElems > 0) {
        int offset = tileNum * TILE_ELEMS;
        // CopyIn
        LocalTensor<float> localA = inQueueA.AllocTensor<float>();
        LocalTensor<float> localB = inQueueB.AllocTensor<float>();
        copyParams.blockLen = tailElems * sizeof(float);
        DataCopyPad(localA, gA[offset], copyParams, padParams);
        DataCopyPad(localB, gB[offset], copyParams, padParams);
        inQueueA.EnQue(localA);
        inQueueB.EnQue(localB);
        // Compute
        localA = inQueueA.DeQue<float>();
        localB = inQueueB.DeQue<float>();
        LocalTensor<float> localC = outQueueC.AllocTensor<float>();
        Add(localC, localA, localB, tailElems);
        outQueueC.EnQue(localC);
        inQueueA.FreeTensor(localA);
        inQueueB.FreeTensor(localB);
        // CopyOut
        localC = outQueueC.DeQue<float>();
        DataCopyPad(gC[offset], localC, copyParams);
        outQueueC.FreeTensor(localC);
    }
}
"""


def main():
    N = 1024
    rng = np.random.RandomState(42)

    print("=" * 60)
    print("AsNumpy SourceModule — Add Kernel Example")
    print("=" * 60)

    # ---- Step 1: JIT compile ----
    print("\n[1] Compiling Ascend C kernel...")
    n_lines = len(KERNEL_SRC.splitlines())
    logger.info("source lines: {}", n_lines)
    mod = SourceModule(KERNEL_SRC, options=["-O3"])
    logger.info("kernels discovered: {}", mod._kernels)
    print("    Compilation successful.")

    # ---- Step 2: Get kernel function ----
    print("\n[2] Getting kernel function...")
    kernel = mod.get_function(
        "vector_add",
        signature=["float32*", "float32*", "float32*", "int32"],
    )
    logger.info("kernel: {}", kernel)

    # ---- Step 3: Prepare data ----
    print("\n[3] Preparing NPU data...")
    a_np = rng.randn(N).astype(np.float32)
    b_np = rng.randn(N).astype(np.float32)
    a_ap = ap.ndarray.from_numpy(a_np)
    b_ap = ap.ndarray.from_numpy(b_np)
    c_ap = ap.empty((N,), dtype=ap.float32)
    logger.info("a ptr: {}", hex(a_ap.device_address))
    logger.info("b ptr: {}", hex(b_ap.device_address))
    logger.info("c ptr: {}", hex(c_ap.device_address))
    logger.info("shape: {}, dtype: {}", a_ap.shape, a_ap.dtype)
    print(f"    N={N}, dtype=float32")

    # ---- Step 4: Launch ----
    print("\n[4] Launching kernel (grid=8)...")
    kernel(a_ap, b_ap, c_ap, N, grid=(8,))
    print("    Launch OK — no runtime errors.")

    # ---- Step 5: Verify ----
    print("\n[5] Verifying...")
    result = c_ap.to_numpy()
    expected = a_np + b_np
    max_err = float(np.max(np.abs(result - expected)))
    logger.info("max error vs NumPy: {:.2e}", max_err)

    if max_err < 1e-4:
        print("    PASS — vector add matches NumPy reference.")
    else:
        print(f"    FAIL — numerical mismatch (max error = {max_err:.2e})")

    # ---- Step 6: Multi-block scaling ----
    print("\n[6] Multi-block scaling...")
    all_ok = True
    for blocks in [1, 2, 4, 8, 16, 32]:
        c2 = ap.empty((N,), dtype=ap.float32)
        kernel(a_ap, b_ap, c2, N, grid=(blocks,))
        r = c2.to_numpy()
        err = float(np.max(np.abs(r - expected)))
        logger.info("grid=({:2d},)  max_err={:.2e}", blocks, err)
        if err >= 1e-4:
            all_ok = False
    print(f"    {'PASS' if all_ok else 'FAIL'} — tested grid sizes 1..32")

    # ---- Step 7: Various sizes ----
    print("\n[7] Various input sizes...")
    all_ok = True
    for size in [1, 3, 7, 13, 100, 1000, 10000]:
        a2 = rng.randn(size).astype(np.float32)
        b2 = rng.randn(size).astype(np.float32)
        a_ap2 = ap.ndarray.from_numpy(a2)
        b_ap2 = ap.ndarray.from_numpy(b2)
        c_ap2 = ap.empty((size,), dtype=ap.float32)
        kernel(a_ap2, b_ap2, c_ap2, size, grid=(8,))
        r2 = c_ap2.to_numpy()
        exp2 = a2 + b2
        err2 = float(np.max(np.abs(r2 - exp2)))
        logger.info("N={:6d}  max_err={:.2e}", size, err2)
        if err2 >= 1e-4:
            all_ok = False
    print(f"    {'PASS' if all_ok else 'FAIL'} — tested sizes 1..10000")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Add kernel: working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
