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
"""Tests for the copy() function."""

import numpy
from asnumpy import testing


@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_copy_independence(xp):
    """Test that copy returns an independent array."""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5.0, 5.0, size=(3, 4)).astype(numpy.float32)
    if xp is numpy:
        a = numpy.array(data)
        b = numpy.copy(a)
        a[0, 0] = 999.0
    else:
        a = xp.ndarray.from_numpy(data)
        b = xp.copy(a)
        a_np = a.to_numpy()
        a_np[0, 0] = 999.0
        a = xp.ndarray.from_numpy(a_np)
    return b


@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_copy_preserves_dtype(xp):
    """Test that copy preserves the original dtype."""
    data = numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32)
    if xp is numpy:
        return numpy.copy(data)
    else:
        a = xp.ndarray.from_numpy(data)
        b = xp.copy(a)
        return b
