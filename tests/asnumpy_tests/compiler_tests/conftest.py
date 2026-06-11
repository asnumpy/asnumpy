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

"""Test fixtures for compiler module tests."""

import os
import pytest
from unittest.mock import MagicMock, patch


# ==========================================================================
# Skip markers
# ==========================================================================

def _bisheng_available() -> bool:
    """Return True if the bisheng compiler can be located."""
    from asnumpy.compiler.source_module import _find_bisheng
    try:
        _find_bisheng()
        return True
    except Exception:
        return False


requires_bisheng = pytest.mark.skipif(
    not _bisheng_available(),
    reason="Requires bisheng compiler",
)

requires_npu = requires_bisheng  # same env check for now

# CANN 9.1 + Ascend 910B4: bisheng generates code that causes AI Core
# "Illegal instruction" (errCode=0x10) for arithmetic ops (Add, Mul, etc).
# Only DataCopy and Duplicate work correctly.
requires_kernel_exec = pytest.mark.skip(
    reason="Bisheng compiler on 910B4 generates illegal AI Core instructions "
    "for arithmetic ops. DataCopy and Duplicate work correctly. "
    "TODO: retest with fixed bisheng compiler."
)


# ==========================================================================
# Kernel source fixtures (verified against CANN 8.2+ Ascend C API)
# ==========================================================================

VECTOR_ADD_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void vector_add(
    __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TPipe pipe;
    pipe.Init();

    TBuf<TPosition::VECIN> buf_a, buf_b;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf_a, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_b, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_out, static_cast<uint32_t>(count));

    GlobalTensor<float> gA, gB, gC;
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

FILL_CONST_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void fill_const(
    __gm__ float* out, float value, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TPipe pipe;
    pipe.Init();

    TBuf<TPosition::VECIN> buf;
    pipe.InitBuffer(buf, static_cast<uint32_t>(count));

    GlobalTensor<float> gOut;
    gOut.SetGlobalBuffer(out + start, static_cast<uint64_t>(count));

    LocalTensor<float> local = buf.AllocTensor<float>();
    Duplicate(local, value, count);
    DataCopy(gOut, local, count);
    buf.FreeTensor(local);
}
"""

MULTI_KERNEL_SOURCE = r"""
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void kernel_one(
    __gm__ float* a, __gm__ float* b, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TPipe pipe;
    pipe.Init();

    TBuf<TPosition::VECIN> buf;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_out, static_cast<uint32_t>(count));

    GlobalTensor<float> gA, gB;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));

    LocalTensor<float> local = buf.AllocTensor<float>();
    DataCopy(local, gA, count);
    Add(local, local, local, count);
    DataCopy(gB, local, count);
    buf.FreeTensor(local);
}

extern "C" __global__ __aicore__ void kernel_two(
    __gm__ float* a, __gm__ float* b, int n)
{
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int per_block = (n + block_num - 1) / block_num;
    int start = block_idx * per_block;
    int count = (per_block < n - start) ? per_block : (n - start);

    TPipe pipe;
    pipe.Init();

    TBuf<TPosition::VECIN> buf;
    TBuf<TPosition::VECOUT> buf_out;
    pipe.InitBuffer(buf, static_cast<uint32_t>(count));
    pipe.InitBuffer(buf_out, static_cast<uint32_t>(count));

    GlobalTensor<float> gA, gB;
    gA.SetGlobalBuffer(a + start, static_cast<uint64_t>(count));
    gB.SetGlobalBuffer(b + start, static_cast<uint64_t>(count));

    LocalTensor<float> local = buf.AllocTensor<float>();
    DataCopy(local, gA, count);
    Mul(local, local, local, count);
    DataCopy(gB, local, count);
    buf.FreeTensor(local);
}
"""


@pytest.fixture
def vector_add_source() -> str:
    return VECTOR_ADD_SOURCE


@pytest.fixture
def fill_const_source() -> str:
    return FILL_CONST_SOURCE


@pytest.fixture
def multi_kernel_source() -> str:
    return MULTI_KERNEL_SOURCE


# ==========================================================================
# Mock fixtures for _rt module
# ==========================================================================

@pytest.fixture
def mock_rt():
    """Return a MagicMock that patches asnumpy.compiler._rt."""
    mock = MagicMock(name="_rt")
    _handle_counter = [1000]

    def _next_handle():
        _handle_counter[0] += 1
        return _handle_counter[0]

    mock.register_binary.return_value = _next_handle()
    mock.register_function.return_value = None
    mock.unregister_binary.return_value = None
    mock.launch_kernel.return_value = None
    return mock


@pytest.fixture
def mock_rt_context():
    """Context manager to patch _rt module.

    Usage:
        with _rt_patch(mock_rt):
            mod = SourceModule(source)
    """
    def _patch():
        return patch.multiple(
            "asnumpy.compiler.source_module._rt",
            register_binary=MagicMock(return_value=1001),
            register_function=MagicMock(return_value=None),
            unregister_binary=MagicMock(return_value=None),
            launch_kernel=MagicMock(return_value=None),
        )

    return _patch
