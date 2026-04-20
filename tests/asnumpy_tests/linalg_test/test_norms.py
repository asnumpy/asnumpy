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

"""线性代数范数与行列式测试

包含：
1. 范数计算: norm
2. 行列式: det
3. 符号行列式: slogdet

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


def _assert_slogdet_allclose(data, rtol=1e-5, atol=1e-5):
    """辅助函数：比较 numpy 和 asnumpy 的 slogdet 结果

    slogdet 返回元组 (sign, logdet)，numpy_asnumpy_allclose 装饰器
    无法正确处理元组返回值，因此使用此函数手动逐项比较。
    """
    import asnumpy as ap

    np_sign, np_logdet = numpy.linalg.slogdet(data)

    a = _create_array(ap, data, data.dtype)
    ap_sign, ap_logdet = ap.linalg.slogdet(a)

    numpy.testing.assert_allclose(ap_sign.to_numpy(), np_sign, rtol=rtol, atol=atol)
    numpy.testing.assert_allclose(ap_logdet.to_numpy(), np_logdet, rtol=rtol, atol=atol)


# ==========================================================================
# 1. 范数测试 (Norm)
# ==========================================================================

# ---------- 1.1 基础功能: 向量范数 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_vector_l2(xp, dtype):
    """向量 L2 范数 (默认)"""
    data = [3.0, 4.0]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_vector_l1(xp, dtype):
    """向量 L1 范数"""
    data = [1.0, -2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, ord=1)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_vector_inf(xp, dtype):
    """向量无穷范数"""
    data = [1.0, -5.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, ord=float('inf'))


# ---------- 1.2 基础功能: 矩阵范数 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_matrix_frobenius(xp, dtype):
    """矩阵 Frobenius 范数 (默认)"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_matrix_frobenius_explicit(xp, dtype):
    """矩阵 Frobenius 范数 (显式 ord='fro')"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, ord='fro')


# ---------- 1.3 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_fp32_precision(xp, dtype):
    """FP32 精度: 范数结果应与 NumPy 一致"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_norm_fp64_precision(xp, dtype):
    """FP64 精度: 范数结果应与 NumPy 一致 (高精度)"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-10, 10, (5,)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_norm_matrix_fp32(xp, dtype):
    """FP32 精度: 矩阵范数"""
    numpy.random.seed(55)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_norm_matrix_fp64(xp, dtype):
    """FP64 精度: 矩阵范数"""
    numpy.random.seed(55)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


# ---------- 1.4 axis 参数 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_axis_0(xp, dtype):
    """axis=0: 沿第 0 轴求范数"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(0,))


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_axis_1(xp, dtype):
    """axis=1: 沿第 1 轴求范数"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(1,))


# ---------- 1.5 keepdims 参数 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_keepdims(xp, dtype):
    """keepdims=True: 保持维度"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(0,), keepdims=True)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_keepdims_axis1(xp, dtype):
    """keepdims=True: 沿 axis=1 保持维度"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(1,), keepdims=True)


# ---------- 1.6 非方阵输入 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_nonsquare_matrix(xp, dtype):
    """非方阵: 矩阵 Frobenius 范数"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_nonsquare_axis(xp, dtype):
    """非方阵: 沿轴求范数"""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(1,))


# ---------- 1.7 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_3d_array(xp, dtype):
    """广播: 3D 数组沿最后两个轴求范数"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (2, 3, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(1, 2))


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_batch_vectors(xp, dtype):
    """广播: 批量向量范数"""
    numpy.random.seed(33)
    data = numpy.random.uniform(-3, 3, (3, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a, axis=(1,))


# ---------- 1.8 空矩阵输入 ----------
@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_norm_empty_vector(xp, dtype):
    """空矩阵: 空向量范数"""
    a = _create_array(xp, [], dtype)
    return xp.linalg.norm(a)


# ---------- 1.9 随机矩阵 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_norm_random_vector(xp, dtype):
    """随机向量: 范数"""
    numpy.random.seed(77)
    data = numpy.random.uniform(-20, 20, (10,)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_norm_random_matrix_fp64(xp, dtype):
    """随机矩阵: FP64 范数"""
    numpy.random.seed(88)
    data = numpy.random.uniform(-10, 10, (5, 5)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.norm(a)


# ==========================================================================
# 2. 行列式测试 (Determinant)
# ==========================================================================

# ---------- 2.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_2x2(xp, dtype):
    """2x2 行列式"""
    data = [[1.0, 2.0], [3.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_3x3(xp, dtype):
    """3x3 行列式"""
    data = [[6.0, 1.0, 1.0],
            [4.0, -2.0, 5.0],
            [2.0, 8.0, 7.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ---------- 2.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_det_fp32_precision(xp, dtype):
    """FP32 精度: 行列式"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_det_fp64_precision(xp, dtype):
    """FP64 精度: 行列式 (高精度)"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ---------- 2.3 特殊矩阵 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_identity(xp, dtype):
    """单位阵: det = 1"""
    data = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_diagonal(xp, dtype):
    """对角阵: det = 对角线乘积"""
    data = [[2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 5.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ---------- 2.4 奇异矩阵边界 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_singular_zero(xp, dtype):
    """奇异矩阵: det = 0"""
    data = [[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_zero_matrix(xp, dtype):
    """奇异矩阵: 全零矩阵 det = 0"""
    data = [[0.0, 0.0],
            [0.0, 0.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_rank_deficient(xp, dtype):
    """奇异矩阵: 行线性相关"""
    data = [[1.0, 2.0],
            [2.0, 4.0]]
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ---------- 2.5 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_batch_matrices(xp, dtype):
    """广播: 批量矩阵行列式"""
    data = numpy.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ]).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float32, numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-4)
def test_det_batch_3x3(xp, dtype):
    """广播: 批量 3x3 行列式"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-3, 3, (2, 3, 3)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ---------- 2.6 空矩阵输入 ----------
@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_det_empty_matrix(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 0)), dtype)
    return xp.linalg.det(a)


# ---------- 2.7 随机矩阵 ----------
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-3, atol=1e-3)
def test_det_random_4x4(xp, dtype):
    """随机矩阵: 4x4 行列式"""
    numpy.random.seed(99)
    data = numpy.random.uniform(-10, 10, (4, 4)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-10, atol=1e-10)
def test_det_random_5x5_fp64(xp, dtype):
    """随机矩阵: 5x5 FP64 行列式"""
    numpy.random.seed(111)
    data = numpy.random.uniform(-10, 10, (5, 5)).astype(dtype)
    a = _create_array(xp, data, dtype)
    return xp.linalg.det(a)


# ==========================================================================
# 3. 符号行列式测试 (Slogdet)
# ==========================================================================

# ---------- 3.1 基础功能 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_2x2(dtype):
    """2x2 slogdet"""
    data = numpy.array([[1.0, 2.0], [3.0, 4.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_3x3(dtype):
    """3x3 slogdet"""
    data = numpy.array([[6.0, 1.0, 1.0],
                        [4.0, -2.0, 5.0],
                        [2.0, 8.0, 7.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


# ---------- 3.2 FP32/FP64 精度验证 ----------
@testing.for_dtypes([numpy.float32])
def test_slogdet_fp32_precision(dtype):
    """FP32 精度: slogdet"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    _assert_slogdet_allclose(data, rtol=1e-4, atol=1e-4)


@testing.for_dtypes([numpy.float64])
def test_slogdet_fp64_precision(dtype):
    """FP64 精度: slogdet (高精度)"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-5, 5, (4, 4)).astype(dtype)
    _assert_slogdet_allclose(data, rtol=1e-10, atol=1e-10)


# ---------- 3.3 特殊矩阵 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_identity(dtype):
    """单位阵: sign=1, logdet=0"""
    data = numpy.array([[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_negative_det(dtype):
    """负行列式: sign=-1"""
    data = numpy.array([[-1.0, 0.0],
                        [0.0, 1.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_diagonal(dtype):
    """对角阵: sign 和 logdet 验证"""
    data = numpy.array([[2.0, 0.0, 0.0],
                        [0.0, 3.0, 0.0],
                        [0.0, 0.0, 5.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


# ---------- 3.4 奇异矩阵边界 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_singular(dtype):
    """奇异矩阵: sign=0, logdet=-inf"""
    data = numpy.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_zero_matrix(dtype):
    """奇异矩阵: 全零矩阵"""
    data = numpy.array([[0.0, 0.0],
                        [0.0, 0.0]], dtype=dtype)
    _assert_slogdet_allclose(data)


# ---------- 3.5 广播行为 ----------
@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_batch_matrices(dtype):
    """广播: 批量矩阵 slogdet"""
    data = numpy.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 0.0], [0.0, 3.0]],
    ]).astype(dtype)
    _assert_slogdet_allclose(data)


@testing.for_dtypes([numpy.float32, numpy.float64])
def test_slogdet_batch_3x3(dtype):
    """广播: 批量 3x3 slogdet"""
    numpy.random.seed(42)
    data = numpy.random.uniform(-3, 3, (2, 3, 3)).astype(dtype)
    _assert_slogdet_allclose(data, rtol=1e-4, atol=1e-4)


# ---------- 3.6 空矩阵输入 ----------
@pytest.mark.xfail(reason="NPU operator does not support empty arrays", strict=True)
@testing.for_dtypes([numpy.float32])
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-5)
def test_slogdet_empty_matrix(xp, dtype):
    """空矩阵: 空输入"""
    a = _create_array(xp, numpy.zeros((0, 0)), dtype)
    return xp.linalg.slogdet(a)


# ---------- 3.7 随机矩阵 ----------
@testing.for_dtypes([numpy.float32])
def test_slogdet_random_4x4(dtype):
    """随机矩阵: 4x4 slogdet"""
    numpy.random.seed(99)
    data = numpy.random.uniform(-10, 10, (4, 4)).astype(dtype)
    _assert_slogdet_allclose(data, rtol=1e-3, atol=1e-3)


@testing.for_dtypes([numpy.float64])
def test_slogdet_random_5x5_fp64(dtype):
    """随机矩阵: 5x5 FP64 slogdet"""
    numpy.random.seed(111)
    data = numpy.random.uniform(-10, 10, (5, 5)).astype(dtype)
    _assert_slogdet_allclose(data, rtol=1e-10, atol=1e-10)
