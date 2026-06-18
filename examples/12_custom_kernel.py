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
Example: Multiple custom kernels in one SourceModule
=====================================================

Compiles two Ascend C kernels from a single source string:

1. **vector_mul** — element-wise multiply: ``c[i] = a[i] * b[i]``
2. **scalar_mul** — scalar multiply: ``y[i] = alpha * x[i]``

Key concepts:
  - Multiple ``extern "C"`` kernels in one compilation unit
  - Passing ``float`` scalar parameters (not just ``int``)
  - Reusing the TQue + DataCopyPad + tiling template
  - One binary, multiple entry points

See ``11_add_kernel.py`` for a simpler single-kernel example.
"""

import numpy as np
import asnumpy as ap
from asnumpy.compiler import SourceModule
from loguru import logger

# ==========================================================================
# Ascend C kernels: vector_mul + scalar_mul
#
# Both follow the official CANN TQue + DataCopyPad + tiling pattern.
# scalar_mul demonstrates float scalar parameter passing via Muls().
# ==========================================================================

KERNEL_SRC = r"""
#include "kernel_operator.h"
using namespace AscendC;

constexpr int TILE_ELEMS = 256;
constexpr int PIPELINE_DEPTH = 2;

// ---------------------------------------------------------------------------
// Element-wise vector multiply: c[i] = a[i] * b[i]
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void vector_mul(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int totalLength)
{
    TPipe pipe;
    uint32_t tileBytes = TILE_ELEMS * sizeof(float);

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueueA, inQueueB;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueueC;
    pipe.InitBuffer(inQueueA, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(inQueueB, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outQueueC, PIPELINE_DEPTH, tileBytes);

    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (totalLength + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < totalLength - start)
        ? per_block : (totalLength - start);

    GlobalTensor<float> gA, gB, gC;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));
    gC.SetGlobalBuffer(c + start, static_cast<uint64_t>(count));

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int tileNum = count / TILE_ELEMS;
    int tailElems = count - tileNum * TILE_ELEMS;

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
        Mul(localC, localA, localB, TILE_ELEMS);
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
        LocalTensor<float> localA = inQueueA.AllocTensor<float>();
        LocalTensor<float> localB = inQueueB.AllocTensor<float>();
        copyParams.blockLen = tailElems * sizeof(float);
        DataCopyPad(localA, gA[offset], copyParams, padParams);
        DataCopyPad(localB, gB[offset], copyParams, padParams);
        inQueueA.EnQue(localA);
        inQueueB.EnQue(localB);
        localA = inQueueA.DeQue<float>();
        localB = inQueueB.DeQue<float>();
        LocalTensor<float> localC = outQueueC.AllocTensor<float>();
        Mul(localC, localA, localB, tailElems);
        outQueueC.EnQue(localC);
        inQueueA.FreeTensor(localA);
        inQueueB.FreeTensor(localB);
        localC = outQueueC.DeQue<float>();
        DataCopyPad(gC[offset], localC, copyParams);
        outQueueC.FreeTensor(localC);
    }
}

// ---------------------------------------------------------------------------
// Scalar multiply: y[i] = alpha * x[i]
// First parameter is a float scalar — demonstrates non-pointer kernel args.
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void scalar_mul(
    float alpha, __gm__ float* x, __gm__ float* y, int totalLength)
{
    TPipe pipe;
    uint32_t tileBytes = TILE_ELEMS * sizeof(float);

    TQue<QuePosition::VECIN, PIPELINE_DEPTH> inQueue;
    TQue<QuePosition::VECOUT, PIPELINE_DEPTH> outQueue;
    pipe.InitBuffer(inQueue, PIPELINE_DEPTH, tileBytes);
    pipe.InitBuffer(outQueue, PIPELINE_DEPTH, tileBytes);

    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (totalLength + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < totalLength - start)
        ? per_block : (totalLength - start);

    GlobalTensor<float> gX, gY;
    gX.SetGlobalBuffer(x + start, static_cast<uint64_t>(count));
    gY.SetGlobalBuffer(y + start, static_cast<uint64_t>(count));

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};

    int tileNum = count / TILE_ELEMS;
    int tailElems = count - tileNum * TILE_ELEMS;

    for (int i = 0; i < tileNum; ++i) {
        int offset = i * TILE_ELEMS;
        // CopyIn
        LocalTensor<float> localX = inQueue.AllocTensor<float>();
        copyParams.blockLen = TILE_ELEMS * sizeof(float);
        DataCopyPad(localX, gX[offset], copyParams, padParams);
        inQueue.EnQue(localX);
        // Compute — Muls() is the scalar-multiply intrinsic
        localX = inQueue.DeQue<float>();
        LocalTensor<float> localY = outQueue.AllocTensor<float>();
        Muls(localY, localX, alpha, TILE_ELEMS);
        outQueue.EnQue(localY);
        inQueue.FreeTensor(localX);
        // CopyOut
        localY = outQueue.DeQue<float>();
        DataCopyPad(gY[offset], localY, copyParams);
        outQueue.FreeTensor(localY);
    }

    if (tailElems > 0) {
        int offset = tileNum * TILE_ELEMS;
        LocalTensor<float> localX = inQueue.AllocTensor<float>();
        copyParams.blockLen = tailElems * sizeof(float);
        DataCopyPad(localX, gX[offset], copyParams, padParams);
        inQueue.EnQue(localX);
        localX = inQueue.DeQue<float>();
        LocalTensor<float> localY = outQueue.AllocTensor<float>();
        Muls(localY, localX, alpha, tailElems);
        outQueue.EnQue(localY);
        inQueue.FreeTensor(localX);
        localY = outQueue.DeQue<float>();
        DataCopyPad(gY[offset], localY, copyParams);
        outQueue.FreeTensor(localY);
    }
}
"""


def _check(condition: bool, msg: str) -> None:
    """Assert-like helper that prints a user-friendly message."""
    if not condition:
        print(f"    FAIL — {msg}")
        raise SystemExit(1)


def test_vector_mul(mod, rng) -> None:
    """Test element-wise multiply: c = a * b."""
    N = 1024
    print("\n--- vector_mul: c[i] = a[i] * b[i] ---")

    kernel = mod.get_function(
        "vector_mul",
        signature=["float32*", "float32*", "float32*", "int32"],
    )
    logger.info("kernel: {}", kernel)

    a_np = rng.randn(N).astype(np.float32)
    b_np = rng.randn(N).astype(np.float32)
    a_ap = ap.ndarray.from_numpy(a_np)
    b_ap = ap.ndarray.from_numpy(b_np)
    c_ap = ap.empty((N,), dtype=ap.float32)
    logger.info("a ptr: {}", hex(a_ap.device_address))
    logger.info("b ptr: {}", hex(b_ap.device_address))
    logger.info("c ptr: {}", hex(c_ap.device_address))

    kernel(a_ap, b_ap, c_ap, N, grid=(8,))
    result = c_ap.to_numpy()
    expected = a_np * b_np
    max_err = float(np.max(np.abs(result - expected)))
    logger.info("N={}, grid=8, max_err={:.2e}", N, max_err)
    _check(max_err < 1e-4, f"max error = {max_err:.2e}")
    print("    N=1024, grid=8: PASS")

    # Multi-block
    for blocks in [1, 2, 4, 8, 16]:
        c2 = ap.empty((N,), dtype=ap.float32)
        kernel(a_ap, b_ap, c2, N, grid=(blocks,))
        err = float(np.max(np.abs(c2.to_numpy() - expected)))
        logger.info("grid=({:2d},) max_err={:.2e}", blocks, err)
        _check(err < 1e-4, f"grid=({blocks},) error = {err:.2e}")
    print("    Multi-block (1..16): PASS")


def test_scalar_mul(mod, rng) -> None:
    """Test scalar multiply: y = alpha * x."""
    N = 2048
    alpha = 3.14159
    print(f"\n--- scalar_mul: y[i] = {alpha} * x[i] ---")

    kernel = mod.get_function(
        "scalar_mul",
        signature=["float32", "float32*", "float32*", "int32"],
    )
    logger.info("kernel: {}", kernel)
    logger.info("alpha={:.6f} (signature: float32 first, then pointers)", alpha)

    x_np = rng.randn(N).astype(np.float32)
    x_ap = ap.ndarray.from_numpy(x_np)
    y_ap = ap.empty((N,), dtype=ap.float32)
    logger.info("x ptr: {}", hex(x_ap.device_address))
    logger.info("y ptr: {}", hex(y_ap.device_address))

    kernel(alpha, x_ap, y_ap, N, grid=(8,))
    result = y_ap.to_numpy()
    expected = alpha * x_np
    max_err = float(np.max(np.abs(result - expected)))
    logger.info("N={}, grid=8, max_err={:.2e}", N, max_err)
    _check(max_err < 1e-4, f"max error = {max_err:.2e}")
    print("    N=2048, grid=8: PASS")

    # Multi-block
    for blocks in [1, 2, 4, 8]:
        y2 = ap.empty((N,), dtype=ap.float32)
        kernel(alpha, x_ap, y2, N, grid=(blocks,))
        err = float(np.max(np.abs(y2.to_numpy() - expected)))
        logger.info("grid=({},) max_err={:.2e}", blocks, err)
        _check(err < 1e-4, f"grid=({blocks},) error = {err:.2e}")
    print("    Multi-block (1..8): PASS")

    # Edge sizes
    for size in [1, 3, 7, 13, 100, 10000]:
        x2 = rng.randn(size).astype(np.float32)
        x_ap2 = ap.ndarray.from_numpy(x2)
        y_ap2 = ap.empty((size,), dtype=ap.float32)
        kernel(alpha, x_ap2, y_ap2, size, grid=(4,))
        exp2 = alpha * x2
        err2 = float(np.max(np.abs(y_ap2.to_numpy() - exp2)))
        logger.info("N={:6d}, grid=4, max_err={:.2e}", size, err2)
        _check(err2 < 1e-4, f"N={size} error = {err2:.2e}")
    print("    Edge sizes (1..10000): PASS")


def main() -> None:
    rng = np.random.RandomState(123)

    print("=" * 60)
    print("AsNumpy SourceModule — Multi-Kernel Example")
    print("=" * 60)

    # ---- Step 1: Compile ----
    print("\n[1] Compiling Ascend C source (2 kernels)...")
    n_lines = len(KERNEL_SRC.splitlines())
    logger.info("source lines: {}", n_lines)
    mod = SourceModule(KERNEL_SRC, options=["-O3"])
    logger.info("kernels discovered: {}", mod._kernels)
    print("    Compilation successful — 1 binary, 2 entry points.")

    # ---- Step 2: Test ----
    print("\n[2] Testing kernels...")
    test_vector_mul(mod, rng)
    test_scalar_mul(mod, rng)

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Both kernels pass.")
    print()
    print("How to write your own kernel:")
    print("  1. Copy the TQue + DataCopyPad + tiling template")
    print("  2. Replace 'Mul'/'Muls' with your operator (Add, Sub, Relu...)")
    print("  3. Adjust TILE_ELEMS for wider dtypes (e.g. double → 128)")
    print("  4. Add to KERNEL_SRC, call get_function(), and launch")
    print("=" * 60)


if __name__ == "__main__":
    main()
