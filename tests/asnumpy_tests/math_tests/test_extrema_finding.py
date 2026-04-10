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

import numpy
from asnumpy import testing


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_maximum(xp, dtype):
    """测试 maximum(x1, x2, dtype=None) - 逐元素最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.maximum(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_minimum(xp, dtype):
    """测试 minimum(x1, x2, dtype=None) - 逐元素最小值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.minimum(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_fmax(xp, dtype):
    """测试 fmax(x1, x2, dtype=None) - 逐元素最大值（忽略NaN）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.fmax(a, b)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_fmin(xp, dtype):
    """测试 fmin(x1, x2, dtype=None) - 逐元素最小值（忽略NaN）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    b = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.fmin(a, b)


@testing.for_all_dtypes(no_complex=True, exclude=[numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_max(xp, dtype):
    """测试 max(a) - 数组最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.max(a)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_max_dim(xp, dtype):
    """测试 max(a, axis, keepdims) - 数组沿特定维度的最大值"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.max(a, axis=1, keepdims=True)


@testing.for_all_dtypes(no_complex=True, exclude=[numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_min(xp, dtype):
    """测试 min(a) - 数组最大值"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.min(a)


@testing.for_all_dtypes(no_complex=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_min_dim(xp, dtype):
    """测试 min(a, axis, keepdims) - 数组沿特定维度的最大值"""
    a = testing.shaped_random((3, 4, 5), dtype=dtype, xp=xp, seed=42)
    return xp.min(a, axis=1, keepdims=True)