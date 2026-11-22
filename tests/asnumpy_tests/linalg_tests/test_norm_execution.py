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

"""Execution-path and dtype tests for the native NPU norm implementation."""

import numpy as np
import pytest
from asnumpy._core.linalg import norm as _native_norm

import asnumpy as ap


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(("axis", "keepdims"), [(None, False), (0, True)])
def test_norm_default_path_does_not_use_host_compute(dtype, axis, keepdims, monkeypatch):
    host = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    expected = np.linalg.norm(host, axis=axis, keepdims=keepdims)

    def host_access_is_forbidden(*args, **kwargs):
        raise AssertionError("default norm must execute entirely on the NPU")

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "norm", host_access_is_forbidden)
        patch.setattr(ap.ndarray, "to_numpy", host_access_is_forbidden)
        result = ap.linalg.norm(ap.asarray(host), axis=axis, keepdims=keepdims)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6, atol=1e-6)


def test_norm_integer_input_uses_npu_and_matches_numpy_dtype(monkeypatch):
    host = np.array([1, 2, 3], dtype=np.int64)

    def host_access_is_forbidden(*args, **kwargs):
        raise AssertionError("integer norm must execute entirely on the NPU")

    with monkeypatch.context() as patch:
        patch.setattr(np.linalg, "norm", host_access_is_forbidden)
        patch.setattr(ap.ndarray, "to_numpy", host_access_is_forbidden)
        result = ap.linalg.norm(ap.asarray(host))

    assert result.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(result.to_numpy(), np.linalg.norm(host))


@pytest.mark.parametrize("keepdims", [False, True])
def test_native_norm_empty_axis_reduces_all_dimensions(keepdims):
    host = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = _native_norm(ap.asarray(host), 2.0, (), keepdims)

    expected = np.linalg.norm(host, keepdims=keepdims)
    assert result.shape == expected.shape
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6, atol=1e-6)
