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

from __future__ import annotations

import numpy as np
import pytest

import asnumpy as ap

from .helpers import assert_numpy_equal


def test_array_creates_asnumpy_ndarray_from_list():
    x = ap.array([1, 2, 3], dtype=np.float32)
    assert isinstance(x, ap.ndarray)
    assert x.dtype == np.dtype("float32")
    assert_numpy_equal(x, np.array([1, 2, 3], dtype=np.float32))


def test_array_from_numpy_array():
    np_x = np.array([1, 2, 3], dtype=np.int32)
    x = ap.array(np_x)
    assert isinstance(x, ap.ndarray)
    assert_numpy_equal(x, np_x)


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.int64, np.float64])
def test_array_creates_from_various_dtypes(dtype):
    x = ap.array([1, 2, 3], dtype=dtype)
    assert x.dtype == np.dtype(dtype)
    assert_numpy_equal(x, np.array([1, 2, 3], dtype=dtype))


def test_array_copy_false_reuses_asnumpy_array():
    x = ap.array([1, 2, 3], dtype=np.float32)
    y = ap.array(x, copy=False)
    assert y is x


def test_asarray_reuses_asnumpy_array_when_dtype_matches():
    x = ap.array([1, 2, 3], dtype=np.int32)
    y = ap.asarray(x)
    assert y is x


def test_asarray_converts_numpy_array_to_asnumpy():
    np_x = np.array([1, 2, 3], dtype=np.int32)
    ap_x = ap.asarray(np_x)
    assert isinstance(ap_x, ap.ndarray)
    assert_numpy_equal(ap_x, np_x)


def test_asarray_dtype_conversion():
    x = ap.array([1, 2, 3], dtype=np.int32)
    y = ap.asarray(x, dtype=np.float32)
    assert y.dtype == np.dtype("float32")
    assert_numpy_equal(y, np.array([1, 2, 3], dtype=np.float32))


def test_asarray_copy_true():
    x = ap.array([1, 2, 3], dtype=np.int32)
    y = ap.asarray(x, copy=True)
    assert isinstance(y, ap.ndarray)
    assert y is not x
    assert_numpy_equal(y, x)


def test_asnumpy_returns_numpy_array():
    x = ap.array([1, 2, 3], dtype=np.float32)
    np_x = ap.asnumpy(x)
    assert isinstance(np_x, np.ndarray)
    np.testing.assert_array_equal(np_x, np.array([1, 2, 3], dtype=np.float32))
