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
"""Tests for the array() creation function."""

import numpy
from asnumpy import testing


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_from_numpy_array(xp, dtype):
    """Test creating array from an existing numpy array."""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5.0, 5.0, size=(3, 4)).astype(dtype)
    if xp is numpy:
        return numpy.array(data)
    return xp.array(data)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_from_list(xp, dtype):
    """Test creating array from a Python list."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    return xp.array(data, dtype=dtype)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_from_nested_list(xp, dtype):
    """Test creating array from a nested Python list."""
    data = [[1.0, 2.0], [3.0, 4.0]]
    return xp.array(data, dtype=dtype)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_from_tuple(xp, dtype):
    """Test creating array from a Python tuple."""
    data = (1.0, 2.0, 3.0)
    return xp.array(data, dtype=dtype)


@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_from_scalar(xp):
    """Test creating array from a scalar."""
    return xp.array(42.0, dtype=numpy.float32)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_with_ndmin(xp, dtype):
    """Test creating array with ndmin parameter."""
    data = [1.0, 2.0, 3.0]
    return xp.array(data, dtype=dtype, ndmin=2)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_array_copy_true(xp, dtype):
    """Test that array() with copy=True produces correct data."""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5.0, 5.0, size=(3,)).astype(dtype)
    if xp is numpy:
        a = numpy.array(data)
        b = numpy.array(a, copy=True)
    else:
        a = xp.ndarray.from_numpy(data)
        b = xp.array(a, copy=True)
    return b
