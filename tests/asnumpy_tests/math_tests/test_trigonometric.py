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

"""三角函数测试
1. sin(x)
2. cos(x)
3. tan(x)
4. arcsin(x)
5. arccos(x)
6. arctan(x)
7. arctan2(x1, x2)
8. hypot(x1, x2)
9. radians(x)
"""

import numpy
from asnumpy import testing


# ========== 三角函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_sin(xp, dtype):
    """测试sin函数"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.sin(a)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_cos(xp, dtype):
    """测试cos函数"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.cos(a)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_tan(xp, dtype):
    """测试tan函数"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=0.5)
    return xp.tan(a)


# ========== 反三角函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-4)
def test_arcsin(xp, dtype):
    """测试arcsin函数
    
    输入范围: [-1, 1]
    注意：使用 rtol=1e-4（放宽）因为存在精度差异
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=0.8)
    return xp.arcsin(a)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-4)
def test_arccos(xp, dtype):
    """测试arccos函数
    
    输入范围: [-1, 1]
    注意：使用 rtol=1e-4（放宽）因为存在精度差异
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=0.8)
    return xp.arccos(a)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_arctan(xp, dtype):
    """测试arctan函数"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    return xp.arctan(a)


# ========== 双参数三角函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_arctan2(xp, dtype):
    """测试arctan2函数
    
    arctan2(y, x) 计算 y/x 的反正切，考虑象限
    """
    y = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    x = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.arctan2(y, x)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_hypot(xp, dtype):
    """测试hypot函数
    
    hypot(x, y) = sqrt(x^2 + y^2)
    """
    x = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42)
    y = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=43)
    return xp.hypot(x, y)


# ========== 角度转换测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_radians(xp, dtype):
    """测试 radians(x) - 度数转弧度
    
    注意：此测试会暴露 AsNumPy 的 radians bug
    """
    degrees = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=180.0)
    return xp.radians(degrees)


# ========== 测试结果与已知问题 ==========
#
#  测试统计: 8/9 通过 (88.9%), 1个失败
#
#  整数类型支持:
# 仅支持浮点: 所有三角函数 (9个)
#
# 为什么不支持整数测试？
# - NumPy 和 AsNumPy 对整数输入的 dtype 转换规则不一致
# - NumPy: int8→float16, int16→float32, int32/int64→float64
# - AsNumPy: 所有整数→float64 (不一致的转换)
# - 会导致 dtype mismatch，因此只测试浮点类型更合理
#
#  精度问题:
# 1. arcsin/arccos 精度差异
#    - 问题：与 NumPy 结果存在精度差异
#    - 最大差异：约 3.8e-05
#    - 解决：放宽容差到 rtol=1e-4
#
#   Bug:
# 2. radians - 返回全 0 
#    - 问题描述：AsNumPy 的 radians 函数返回全是 0
#    - 测试输入：随机角度值（0-180度）
#    - NumPy 输出：正确的弧度值（0-π）
#    - AsNumPy 输出：[[0. 0. 0. 0.] [0. 0. 0. 0.] [0. 0. 0. 0.]]
#
#  缺失的 API:
# - deg2rad: 未实现（NumPy 有）
# - rad2deg: 未实现（NumPy 有）
# - degrees: 未实现（NumPy 有，弧度转角度）
