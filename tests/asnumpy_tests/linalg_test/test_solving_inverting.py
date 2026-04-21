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

"""矩阵求逆与求解函数测试

包含：
1. 矩阵求逆: inv

优化维度：
- FP32/FP64 精度验证
- 奇异矩阵边界
- 非方阵输入
- 广播行为
- 空矩阵输入
"""

import numpy
import pytest
from asnumpy import testing


# ========== 辅助函数 ==========
def _create_array(xp, data, dtype):
    """辅助函数：创建数组"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ==========================================================================
# 1. 矩阵求逆测试 (Matrix Inverse)
# ==========================================================================

# ---------- 1.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_2x2(xp, dtype):
    """2x2 矩阵求逆"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_3x3(xp, dtype):
    """3x3 矩阵求逆"""
    data = [[1.0, 2.0, 3.0],
            [0.0, 1.0, 4.0],
            [5.0, 6.0, 0.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_4x4(xp, dtype):
    """4x4 矩阵求逆"""
    data = [[2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_fp32_precision(xp, dtype):
    """FP32 精度: 矩阵求逆"""
    numpy.random.seed(42)
    # 生成可逆矩阵: 随机矩阵大概率可逆
    data = numpy.random.uniform(-5, 5, (3, 3)).astype(dtype)
    # 确保对角占优以增强数值稳定性
    data = data + numpy.eye(3, dtype=dtype) * 5.0
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-8, atol=1e-8)
def test_inv_fp64_precision(xp, dtype):
    """FP64 精度: 矩阵求逆 (高精度)"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (3, 3)).astype(dtype)
    data = data + numpy.eye(3, dtype=dtype) * 5.0
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.3 特殊矩阵 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_identity(xp, dtype):
    """单位阵: 逆 = 自身"""
    data = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_diagonal(xp, dtype):
    """对角阵: 逆 = 对角元素倒数"""
    data = [[2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 5.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.4 奇异矩阵边界 ----------
@pytest.mark.xfail(reason="Singular matrix has no inverse", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_singular(xp, dtype):
    """奇异矩阵: 行线性相关，应失败"""
    data = [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@pytest.mark.xfail(reason="Singular matrix has no inverse", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_zero_matrix(xp, dtype):
    """奇异矩阵: 全零矩阵，应失败"""
    data = [[0.0, 0.0],
            [0.0, 0.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.5 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_batch_matrices(xp, dtype):
    """广播: 批量矩阵求逆"""
    data = numpy.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ]).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_batch_3x3(xp, dtype):
    """广播: 批量 3x3 矩阵求逆"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-3, 3, (2, 3, 3)).astype(dtype)
    # 对角占优确保可逆
    for i in range(2):
        data[i] = data[i] + numpy.eye(3, dtype=dtype) * 5.0
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.6 非方阵输入 ----------
@pytest.mark.xfail(reason="Non-square matrix has no inverse", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_nonsquare(xp, dtype):
    """非方阵: 应失败"""
    data = [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


# ---------- 1.7 空矩阵输入 ----------
@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_inv_empty_matrix(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 0)), dtype)
    return xp.linalg.inv(a)


# ---------- 1.8 随机矩阵 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-2, atol=1e-2)
def test_inv_random_4x4(xp, dtype):
    """随机矩阵: 4x4 求逆"""
    numpy.random.seed(99)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    # 对角占优
    data = data + numpy.eye(4, dtype=dtype) * 10.0
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-8, atol=1e-8)
def test_inv_random_5x5_fp64(xp, dtype):
    """随机矩阵: 5x5 FP64 求逆"""
    numpy.random.seed(111)
    data = numpy.random.uniform(-5, 5, (5, 5)).astype(dtype)
    data = data + numpy.eye(5, dtype=dtype) * 10.0
    a = _create_array(xp, data, dtype)
    return xp.linalg.inv(a)
