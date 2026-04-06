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
"""Tests for the asarray() and asanyarray() creation functions."""

import numpy
from asnumpy import testing


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_asarray_from_list(xp, dtype):
    """Test asarray converts list to NPU array."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    return xp.asarray(data, dtype=dtype)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_asanyarray_from_list(xp, dtype):
    """Test asanyarray converts list to NPU array."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    return xp.asanyarray(data, dtype=dtype)


@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_asarray_no_copy(xp):
    """Test asarray does not copy an existing array of matching dtype."""
    data = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    if xp is numpy:
        a = numpy.asarray(data)
        b = numpy.asarray(data)
    else:
        a = xp.asarray(data)
        b = xp.asarray(data)
    return a
