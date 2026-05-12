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

"""随机分布函数测试

包含：
1. 输出 shape / dtype: pareto, rayleigh, normal, uniform, standard_normal,
   standard_cauchy, weibull, binomial, exponential, geometric, gumbel,
   laplace, logistic, lognormal
2. seed 可复现性占位
3. 边界参数: size=0、负数/非法参数
4. 特殊边界: binomial(n=0)

优化维度：
- 接口 shape / dtype 行为
- 显式参数校验
- 空数组边界
- 后端 seed 能力现状记录

说明：
- 本文件不做统计性质检验。
- 随机结果不与 NumPy 逐元素比较，只验证稳定的接口行为。
"""

import numpy
import pytest

import asnumpy as ap
from asnumpy import testing


# ========== 辅助函数 ==========
def _to_numpy(x):
    """辅助函数：统一转换为 NumPy 数组"""
    if isinstance(x, numpy.ndarray):
        return x
    return x.to_numpy()


def _assert_shape_dtype(result, expected_shape, expected_dtype):
    """辅助函数：断言输出 shape 和 dtype"""
    np_result = _to_numpy(result)
    assert np_result.shape == expected_shape
    assert np_result.dtype == numpy.dtype(expected_dtype)


# ==========================================================================
# 1. 输出 shape / dtype 测试
# ==========================================================================

@pytest.mark.parametrize(
    "func, kwargs, expected_dtype",
    [
        (ap.random.pareto, {"a": 2.5, "size": (2, 3)}, numpy.float32),
        (ap.random.rayleigh, {"scale": 1.5, "size": (2, 3)}, numpy.float32),
        (ap.random.normal, {"loc": 0.0, "scale": 1.0, "size": (2, 3)}, numpy.float64),
        (ap.random.uniform, {"low": -1.0, "high": 1.0, "size": (2, 3)}, numpy.float64),
        (ap.random.standard_normal, {"size": (2, 3)}, numpy.float64),
        (ap.random.standard_cauchy, {"size": (2, 3)}, numpy.float64),
        (ap.random.weibull, {"a": 1.5, "size": (2, 3)}, numpy.float32),
        (ap.random.binomial, {"n": 5, "p": 0.3, "size": (2, 3)}, numpy.int32),
        (ap.random.exponential, {"scale": 1.2, "size": (2, 3)}, numpy.float32),
        (ap.random.geometric, {"p": 0.4, "size": (2, 3)}, numpy.float32),
        (ap.random.gumbel, {"loc": 0.0, "scale": 2.0, "size": (2, 3)}, numpy.float32),
        (ap.random.laplace, {"loc": 0.0, "scale": 1.0, "size": (2, 3)}, numpy.float32),
        (ap.random.logistic, {"loc": 0.0, "scale": 1.0, "size": (2, 3)}, numpy.float32),
        (ap.random.lognormal, {"mean": 0.0, "sigma": 1.0, "size": (2, 3)}, numpy.float32),
    ],
)
def test_distribution_shape_dtype(func, kwargs, expected_dtype):
    """测试各随机分布输出的 shape / dtype"""
    result = func(**kwargs)
    _assert_shape_dtype(result, (2, 3), expected_dtype)


@pytest.mark.parametrize(
    "func, kwargs, expected_dtype",
    [
        (ap.random.uniform, {"low": 0.0, "high": 1.0, "size": 5}, numpy.float64),
        (ap.random.standard_normal, {"size": 5}, numpy.float64),
        (ap.random.binomial, {"n": 8, "p": 0.5, "size": 5}, numpy.int32),
        (ap.random.exponential, {"scale": 2.0, "size": 5}, numpy.float32),
    ],
)
def test_distribution_shape_dtype_vector_size(func, kwargs, expected_dtype):
    """测试一维 size 输入的 shape / dtype"""
    result = func(**kwargs)
    _assert_shape_dtype(result, (5,), expected_dtype)


# ==========================================================================
# 2. seed 可复现性测试
# ==========================================================================

@pytest.mark.xfail(
    reason="random seed API is not exposed and backend seed handling is inconsistent",
    strict=True,
)
def test_random_seed_reproducibility_placeholder():
    """seed 可复现性占位测试

    当前随机模块未公开统一 seed 接口，且后端不同分布的 seed 策略不一致，
    因此这里保留占位测试，等待后端补齐用户级 seed 能力后再转为稳定测试。
    """
    assert hasattr(ap.random, "seed")


# ==========================================================================
# 3. size=0 边界测试
# ==========================================================================

def test_uniform_empty_size():
    """空数组: uniform 的 size=0"""
    result = ap.random.uniform(0.0, 1.0, 0)
    _assert_shape_dtype(result, (0,), numpy.float64)


def test_normal_empty_size_2d():
    """空数组: normal 的二维空 shape"""
    result = ap.random.normal(0.0, 1.0, (0, 3))
    _assert_shape_dtype(result, (0, 3), numpy.float64)


# ==========================================================================
# 4. 非法参数测试
# ==========================================================================

@pytest.mark.parametrize("a", [0.0, -1.0])
def test_pareto_invalid_a(a):
    """测试 pareto 的非法 a 参数"""
    with pytest.raises(RuntimeError, match=r"pareto"):
        ap.random.pareto(a, (2, 3))


@pytest.mark.parametrize(
    "kwargs, expected_message",
    [
        ({"n": -1, "p": 0.5, "size": (2, 3)}, r"Binomial: n="),
        ({"n": 3, "p": -0.1, "size": (2, 3)}, r"Binomial: p="),
        ({"n": 3, "p": 1.1, "size": (2, 3)}, r"Binomial: p="),
    ],
)
def test_binomial_invalid_parameters(kwargs, expected_message):
    """测试 binomial 的非法参数"""
    with pytest.raises(RuntimeError, match=expected_message):
        ap.random.binomial(**kwargs)


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_exponential_invalid_scale(scale):
    """测试 exponential 的非法 scale 参数"""
    with pytest.raises(RuntimeError, match=r"Exponential: scale="):
        ap.random.exponential(scale, (2, 3))


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1])
def test_geometric_invalid_p(p):
    """测试 geometric 的非法 p 参数"""
    with pytest.raises(RuntimeError, match=r"Geometric: p="):
        ap.random.geometric(p, (2, 3))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_gumbel_invalid_scale(scale):
    """测试 gumbel 的非法 scale 参数"""
    with pytest.raises(RuntimeError, match=r"Gumbel: scale="):
        ap.random.gumbel(0.0, scale, (2, 3))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_laplace_invalid_scale(scale):
    """测试 laplace 的非法 scale 参数"""
    with pytest.raises(RuntimeError, match=r"Laplace: scale="):
        ap.random.laplace(0.0, scale, (2, 3))


@pytest.mark.parametrize("scale", [0.0, -1.0])
def test_logistic_invalid_scale(scale):
    """测试 logistic 的非法 scale 参数"""
    with pytest.raises(RuntimeError, match=r"Logistic: scale="):
        ap.random.logistic(0.0, scale, (2, 3))


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_lognormal_invalid_sigma(sigma):
    """测试 lognormal 的非法 sigma 参数"""
    with pytest.raises(RuntimeError, match=r"Lognormal: sigma="):
        ap.random.lognormal(0.0, sigma, (2, 3))


# ==========================================================================
# 5. 特殊边界行为
# ==========================================================================

def test_binomial_zero_trials_returns_zeros():
    """特殊边界: binomial(n=0) 应返回全 0 数组"""
    result = ap.random.binomial(0, 0.5, (2, 3))
    np_result = _to_numpy(result)
    _assert_shape_dtype(result, (2, 3), numpy.int32)
    numpy.testing.assert_array_equal(np_result, numpy.zeros((2, 3), dtype=numpy.int32))
