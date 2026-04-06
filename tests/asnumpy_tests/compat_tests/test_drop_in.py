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
"""End-to-end tests verifying asnumpy works as a drop-in numpy replacement."""

import numpy
import asnumpy as ap


class TestDropInBasic:
    """Verify the basic 'import asnumpy as np' pattern works."""

    def test_array_create_and_compute(self):
        """Create array and apply a math operation."""
        a = ap.array([1.0, 2.0, 3.0], dtype=numpy.float32)
        result = ap.sin(a)
        expected = numpy.sin(numpy.array([1.0, 2.0, 3.0], dtype=numpy.float32))
        numpy.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)

    def test_zeros_and_add(self):
        """Create zeros and add two arrays."""
        a = ap.zeros((3, 4), dtype=numpy.float32)
        b = ap.ones((3, 4), dtype=numpy.float32)
        c = ap.add(a, b)
        expected = numpy.ones((3, 4), dtype=numpy.float32)
        numpy.testing.assert_allclose(c.to_numpy(), expected, rtol=1e-5)

    def test_array_from_nested_list(self):
        """Create array from nested list and compute."""
        data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        a = ap.array(data, dtype=numpy.float32)
        assert tuple(a.shape) == (2, 3)
        result = ap.mean(a)
        expected = numpy.mean(numpy.array(data, dtype=numpy.float32))
        numpy.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_dtype_available(self):
        """Verify common dtype types are accessible."""
        assert ap.float32 is numpy.float32
        assert ap.int32 is numpy.int32

    def test_constants_available(self):
        """Verify common constants are accessible."""
        assert ap.pi == numpy.pi
        assert ap.inf == numpy.inf
