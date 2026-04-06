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
"""Tests for NumPy-compatible constants and dtype exports."""

import numpy
import asnumpy as ap


class TestConstants:
    """Test that NumPy constants are accessible from asnumpy."""

    def test_pi(self):
        assert ap.pi == numpy.pi

    def test_e(self):
        assert ap.e == numpy.e

    def test_inf(self):
        assert ap.inf == numpy.inf

    def test_nan(self):
        assert numpy.isnan(ap.nan)

    def test_newaxis(self):
        assert ap.newaxis is None

    def test_euler_gamma(self):
        assert ap.euler_gamma == numpy.euler_gamma


class TestDtypeTypes:
    """Test that NumPy dtype types are accessible from asnumpy."""

    def test_float32(self):
        assert ap.float32 == numpy.float32

    def test_float64(self):
        assert ap.float64 == numpy.float64

    def test_int32(self):
        assert ap.int32 == numpy.int32

    def test_int64(self):
        assert ap.int64 == numpy.int64

    def test_bool(self):
        assert ap.bool_ == numpy.bool_

    def test_dtype_function(self):
        assert ap.dtype('float32') == numpy.dtype('float32')

    def test_finfo(self):
        assert ap.finfo(numpy.float32).max == numpy.finfo(numpy.float32).max

    def test_iinfo(self):
        assert ap.iinfo(numpy.int32).max == numpy.iinfo(numpy.int32).max


class TestDtypeHelpers:
    """Test that NumPy dtype helper functions are accessible from asnumpy."""

    def test_issubdtype(self):
        assert ap.issubdtype(numpy.float32, numpy.floating)

    def test_promote_types(self):
        assert ap.promote_types(numpy.float32, numpy.float64) == numpy.float64

    def test_can_cast(self):
        assert ap.can_cast(numpy.float32, numpy.float64)

    def test_result_type(self):
        assert ap.result_type(numpy.float32, numpy.float64) == numpy.float64
