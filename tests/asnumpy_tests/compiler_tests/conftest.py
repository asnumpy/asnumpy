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

import pytest
from unittest.mock import MagicMock


# ==========================================================================
# Kernel source fixtures (verified against CANN 8.2.RC1 Ascend C API)
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
# Mock fixtures — simulate asnumpy._core.compiler C extension for unit tests
# ==========================================================================

@pytest.fixture
def mock_compiler_lib():
    """Return a MagicMock that simulates the asnumpy._core.compiler module.

    Provides fake implementations for all binary loading, kernel launching,
    event, and stream functions.  Each call generates a unique handle.
    """
    mock = MagicMock(name="compiler")
    _handle_counter = [1000]

    def _next_handle():
        _handle_counter[0] += 1
        return _handle_counter[0]

    # Binary lifecycle (ACL path)
    mock.load_binary.return_value = _next_handle()
    mock.unload_binary.return_value = None
    mock.get_function.return_value = _next_handle()

    # Binary lifecycle (RTS path — registered via _rts_loader, not C++)
    mock.register_binary.return_value = _next_handle()
    mock.register_function.return_value = None
    mock.unregister_binary.return_value = None

    # Kernel launch
    mock.launch_kernel.return_value = None
    mock.launch_kernel_rts.return_value = None

    # Events
    mock.create_event.return_value = _next_handle()
    mock.destroy_event.return_value = None
    mock.record_event.return_value = None
    mock.synchronize_event.return_value = None
    mock.elapsed_time_between.return_value = 1.5

    # Streams
    mock.create_stream.return_value = _next_handle()
    mock.destroy_stream.return_value = None
    mock.synchronize_stream.return_value = None

    # Error
    mock.get_last_error.return_value = ""

    return mock


@pytest.fixture
def mock_rts_loader():
    """Return a MagicMock that simulates the _rts_loader module."""
    mock = MagicMock(name="_rts_loader")
    _handle_counter = [2000]

    def _next_handle():
        _handle_counter[0] += 1
        return _handle_counter[0]

    mock.register_binary.return_value = _next_handle()
    mock.register_function.return_value = None
    mock.unregister_binary.return_value = None
    mock.launch_kernel.return_value = None

    return mock


@pytest.fixture(autouse=True)
def _clean_atexit_registry():
    """Clear SourceModule's atexit registry before and after each test.

    Prevents stale SourceModule instances (from failed tests) from
    attempting cleanup in subsequent tests.
    """
    from asnumpy.compiler.source_module import _atexit_registry
    _atexit_registry.clear()
    yield
    _atexit_registry.clear()
