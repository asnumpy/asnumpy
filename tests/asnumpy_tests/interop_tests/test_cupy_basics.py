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

"""End-to-end coverage based on CuPy's Basics user guide.

Reference: docs/source/user_guide/basic.rst in cupy/cupy, including the
``array``/``asarray`` device-transfer and ``asnumpy`` host-transfer examples.
"""

import numpy as np

import asnumpy as ap


def test_array_creates_device_array_from_sequence():
    result = ap.array([1, 2, 3])

    assert isinstance(result, ap.ndarray)
    assert result.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(ap.asnumpy(result), np.array([1, 2, 3]))


def test_array_copies_device_input_by_default():
    source = ap.asarray(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    result = ap.array(source)

    assert result is not source
    np.testing.assert_array_equal(ap.asnumpy(result), ap.asnumpy(source))


def test_asarray_reuses_device_array_when_dtype_is_unchanged():
    source = ap.array([1.0, 2.0, 3.0], dtype=np.float32)

    assert ap.asarray(source) is source
    assert ap.asarray(source, dtype=np.float32) is source


def test_asarray_casts_device_array_when_dtype_changes():
    source = ap.array([1, 2, 3], dtype=np.int32)

    result = ap.asarray(source, dtype=np.float32)

    assert result is not source
    assert result.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(ap.asnumpy(result), np.array([1, 2, 3], dtype=np.float32))


def test_asarray_accepts_noncontiguous_numpy_input():
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]

    result = ap.asarray(source)

    np.testing.assert_array_equal(ap.asnumpy(result), source)


def test_asnumpy_accepts_host_input():
    source = np.array([1, 2, 3])

    assert ap.asnumpy(source) is source
    np.testing.assert_array_equal(ap.asnumpy([1, 2, 3]), source)


def test_ndarray_get_copies_to_host():
    source = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = ap.asarray(source).get()

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, source)


def test_get_array_module_selects_backend():
    host = np.array([1, 2, 3])
    device = ap.asarray(host)

    assert ap.get_array_module(host) is np
    assert ap.get_array_module([1, 2, 3]) is np
    assert ap.get_array_module(device) is ap
    assert ap.get_array_module(host, device) is ap


def test_cupy_basics_host_device_roundtrip():
    """Run creation, H2D transfer, NPU arithmetic, and D2H transfer together."""
    shape = (1024, 1024)
    x_cpu = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y_cpu = np.full(shape, 2.0, dtype=np.float32)

    # Access asarray first to cover the same-named array module/function export.
    x_npu = ap.asarray(x_cpu)
    y_npu = ap.array(y_cpu)
    result_npu = x_npu + y_npu
    result_cpu = ap.asnumpy(result_npu)

    assert isinstance(x_npu, ap.ndarray)
    assert isinstance(y_npu, ap.ndarray)
    assert isinstance(result_npu, ap.ndarray)
    assert isinstance(result_cpu, np.ndarray)
    np.testing.assert_array_equal(result_cpu, x_cpu + y_cpu)
