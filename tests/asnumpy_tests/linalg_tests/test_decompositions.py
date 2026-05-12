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

"""矩阵分解函数测试

包含：
1. QR 分解: qr

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


def _assert_qr_allclose(data, mode="reduced", rtol=1e-5, atol=1e-5):
    """辅助函数：比较 numpy 和 asnumpy 的 QR 分解结果

    QR 分解返回元组 (Q, R)，numpy_asnumpy_allclose 装饰器
    无法正确处理元组返回值，因此使用此函数手动逐项比较。
    """
    import asnumpy as ap

    # NumPy 结果
    np_result = numpy.linalg.qr(data, mode=mode)

    # AsNumPy 结果
    a = _create_array(ap, data, data.dtype)
    ap_result = ap.linalg.qr(a, mode=mode)

    if mode == "r":
        numpy.testing.assert_allclose(ap_result.to_numpy(), np_result, rtol=rtol, atol=atol)
    else:
        np_q, np_r = np_result
        ap_q, ap_r = ap_result
        numpy.testing.assert_allclose(ap_q.to_numpy(), np_q, rtol=rtol, atol=atol)
        numpy.testing.assert_allclose(ap_r.to_numpy(), np_r, rtol=rtol, atol=atol)


def _assert_qr_factorization(data, mode="reduced", rtol=1e-5, atol=1e-5):
    """辅助函数：验证 QR 分解结构正确性

    奇异或秩亏矩阵的 QR 分解不唯一，零空间对应的正交基可能因实现
    不同而不同，因此不应逐元素比较 Q。这里验证 QR 分解的不变量：
    Q @ R 能重构输入、Q 正交、R 为上三角。
    """
    import asnumpy as ap

    a = _create_array(ap, data, data.dtype)
    q, r = ap.linalg.qr(a, mode=mode)
    q = q.to_numpy()
    r = r.to_numpy()

    numpy.testing.assert_allclose(q @ r, data, rtol=rtol, atol=atol)
    identity = numpy.eye(q.shape[-1], dtype=data.dtype)
    numpy.testing.assert_allclose(q.T @ q, identity, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(numpy.tril(r, k=-1), 0, rtol=rtol, atol=atol)


# ==========================================================================
# 1. QR 分解测试 (QR Decomposition)
# ==========================================================================

# ---------- 1.1 基础功能: reduced 模式 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_reduced_square(dtype):
    """测试 QR 分解 reduced 模式 - 方阵"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-10, 10, (3, 3)).astype(dtype)
    _assert_qr_allclose(data, mode="reduced")


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_reduced_4x4(dtype):
    """测试 QR 分解 reduced 模式 - 4x4 矩阵"""
    numpy.random.seed(123)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    _assert_qr_allclose(data, mode="reduced")


# ---------- 1.2 基础功能: complete 模式 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_complete_square(dtype):
    """测试 QR 分解 complete 模式 - 方阵"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-10, 10, (3, 3)).astype(dtype)
    _assert_qr_allclose(data, mode="complete")


# ---------- 1.3 基础功能: r 模式 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_qr_r_mode(xp, dtype):
    """测试 QR 分解 r 模式 - 仅返回 R 矩阵"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-10, 10, (3, 3)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.qr(a, mode="r")


# ---------- 1.4 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
def test_qr_fp32_precision(dtype):
    """FP32 精度: QR 分解结果应与 NumPy 一致"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.1]], dtype=dtype)
    _assert_qr_allclose(data)


@testing.for_dtypes([numpy.float64])
def test_qr_fp64_precision(dtype):
    """FP64 精度: QR 分解结果应与 NumPy 一致 (高精度)"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.1]], dtype=dtype)
    _assert_qr_allclose(data, rtol=1e-10, atol=1e-10)


# ---------- 1.5 非方阵输入 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_tall_matrix(dtype):
    """非方阵: 高矩阵 (m > n)"""
    data = numpy.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]], dtype=dtype)
    _assert_qr_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_wide_matrix(dtype):
    """非方阵: 宽矩阵 (m < n)"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=dtype)
    _assert_qr_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_tall_complete(dtype):
    """非方阵: 高矩阵 complete 模式"""
    data = numpy.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]], dtype=dtype)
    _assert_qr_allclose(data, mode="complete")


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_wide_complete(dtype):
    """非方阵: 宽矩阵 complete 模式"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]], dtype=dtype)
    _assert_qr_allclose(data, mode="complete")


# ---------- 1.6 奇异矩阵边界 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_singular_matrix(dtype):
    """奇异矩阵: 秩亏矩阵的 QR 分解"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 8.0, 12.0],
                        [7.0, 14.0, 21.0]], dtype=dtype)
    _assert_qr_factorization(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_zero_matrix(dtype):
    """奇异矩阵: 全零矩阵的 QR 分解"""
    data = numpy.array([[0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0]], dtype=dtype)
    _assert_qr_factorization(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_rank_deficient(dtype):
    """奇异矩阵: 部分线性相关行"""
    data = numpy.array([[1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=dtype)
    _assert_qr_factorization(data)


# ---------- 1.7 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_batch_matrices(dtype):
    """广播: 批量矩阵 (batch QR)"""
    data = numpy.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ]).astype(dtype)
    _assert_qr_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_batch_tall(dtype):
    """广播: 批量高矩阵"""
    data = numpy.array([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ]).astype(dtype)
    _assert_qr_allclose(data)


# ---------- 1.8 空矩阵输入 ----------
@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_qr_empty_matrix(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 3)), dtype)
    return xp.linalg.qr(a)


# ---------- 1.9 默认模式 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_qr_default_mode(dtype):
    """默认 mode='reduced'"""
    numpy.random.seed(99)
    data = numpy.random.uniform(-5, 5, (4, 3)).astype(dtype)
    _assert_qr_allclose(data)


# ---------- 1.10 随机矩阵 ----------
@testing.for_dtypes([numpy.float32])
def test_qr_random_5x5(dtype):
    """随机矩阵: 5x5"""
    numpy.random.seed(77)
    data = numpy.random.uniform(-20, 20, (5, 5)).astype(dtype)
    _assert_qr_allclose(data, rtol=1e-4, atol=1e-4)


@testing.for_dtypes([numpy.float64])
def test_qr_random_6x4_fp64(dtype):
    """随机矩阵: 6x4 FP64"""
    numpy.random.seed(88)
    data = numpy.random.uniform(-20, 20, (6, 4)).astype(dtype)
    _assert_qr_allclose(data, rtol=1e-10, atol=1e-10)
