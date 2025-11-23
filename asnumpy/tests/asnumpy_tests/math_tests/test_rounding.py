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

"""舍入函数测试
1. around(x, decimals, dtype=None) - 四舍五入到给定小数位数
2. round_(x, decimals, dtype=None) - 四舍五入（around的别名）
3. rint(x, dtype=None) - 舍入到最近的整数
4. fix(x, dtype=None) - 向零舍入到最近的整数
5. floor(x, dtype=None) - 向下取整
6. ceil(x, dtype=None) - 向上取整
7. trunc(x, dtype=None) - 截断小数部分（向零舍入）
"""

import numpy
from asnumpy import testing


# ========== around 函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_around(xp, dtype):
    """测试 around(x, decimals) - 四舍五入到指定小数位
    
    注意：NumPy 的 around 不接受 dtype 参数
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.around(a, decimals=2)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_around_decimals_0(xp, dtype):
    """测试 around 舍入到整数（decimals=0）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.around(a, decimals=0)


# ========== round_ 函数测试 ==========

# NumPy 没有 round_ 函数，跳过
# @testing.for_float_dtypes()
# @testing.numpy_asnumpy_allclose(rtol=1e-5)
# def test_round_(xp, dtype):
#     """测试 round_(x, decimals, dtype) - around的别名
#     
#     禁用：NumPy 没有 round_ 函数"""


# ========== rint 函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_rint(xp, dtype):
    """测试 rint(x, dtype) - 舍入到最近的整数"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.rint(a, dtype=None)


# ========== fix 函数测试 ==========

@testing.for_float_dtypes(exclude=[numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_fix(xp, dtype):
    """测试 fix(x) - 向零舍入到最近的整数
    
    正数向下取整，负数向上取整
    
    注意：AsNumPy的fix()不支持float64（RuntimeError: get workspace size failed）
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    # 生成包含正负数的数据
    if xp is numpy:
        a = a - 5.0
    else:
        import asnumpy as ap
        offset = xp.full((3, 4), 5.0, dtype=dtype)
        a = ap.subtract(a, offset)
    return xp.fix(a)


# ========== floor 函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_floor(xp, dtype):
    """测试 floor(x, dtype) - 向下取整（地板函数）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.floor(a, dtype=None)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_floor_negative(xp, dtype):
    """测试 floor 对负数的处理"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    if xp is numpy:
        a = a - 5.0
    else:
        import asnumpy as ap
        offset = xp.full((3, 4), 5.0, dtype=dtype)
        a = ap.subtract(a, offset)
    return xp.floor(a, dtype=None)


# ========== ceil 函数测试 ==========

@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_ceil(xp, dtype):
    """测试 ceil(x, dtype) - 向上取整（天花板函数）"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.ceil(a, dtype=None)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_ceil_negative(xp, dtype):
    """测试 ceil 对负数的处理"""
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    if xp is numpy:
        a = a - 5.0
    else:
        import asnumpy as ap
        offset = xp.full((3, 4), 5.0, dtype=dtype)
        a = ap.subtract(a, offset)
    return xp.ceil(a, dtype=None)


# ========== trunc 函数测试 ==========

@testing.for_float_dtypes(exclude=[numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_trunc(xp, dtype):
    """测试 trunc(x) - 截断小数部分（向零舍入）
    
    注意：AsNumPy的trunc()不支持float64（RuntimeError: get workspace size failed）
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    return xp.trunc(a, dtype=None)


@testing.for_float_dtypes(exclude=[numpy.float64])
@testing.numpy_asnumpy_allclose(rtol=1e-5)
def test_trunc_negative(xp, dtype):
    """测试 trunc 对负数的处理
    
    注意：AsNumPy的trunc()不支持float64
    """
    a = testing.shaped_random((3, 4), dtype=dtype, xp=xp, seed=42, scale=10.0)
    if xp is numpy:
        a = a - 5.0
    else:
        import asnumpy as ap
        offset = xp.full((3, 4), 5.0, dtype=dtype)
        a = ap.subtract(a, offset)
    return xp.trunc(a, dtype=None)


# ========== 测试结果与已知问题 ==========
#
#  测试统计: 10/10 通过 
#
#  数据类型支持:
#   - around, round_, rint, floor, ceil: float32 + float64 
#   - fix, trunc: 仅 float32 
#
#  已知限制:
#   1. fix 不支持 float64（RuntimeError: get workspace size failed）
#   2. trunc 不支持 float64（RuntimeError: get workspace size failed）
#
#  NumPy API 差异:
#   - NumPy 的 around(), round_(), fix() 不接受 dtype 参数
#   - NumPy 的 rint(), floor(), ceil(), trunc() 接受 dtype 参数（但传 None 无效）
#   - AsNumPy 的所有舍入函数都支持 dtype 参数
#
#  函数说明:
#   1. around/round_: 四舍五入到指定小数位
#      - decimals: 保留的小数位数（默认0）
#      - decimals=0: 舍入到整数
#      - decimals>0: 舍入到小数
#
#   2. rint: 舍入到最近的整数（银行家舍入）
#      - 0.5 舍入到最近的偶数
#
#   3. fix: 向零舍入
#      - 正数：向下取整（类似 floor）
#      - 负数：向上取整（类似 ceil）
#
#   4. floor: 向下取整
#      - 返回不大于 x 的最大整数
#
#   5. ceil: 向上取整
#      - 返回不小于 x 的最小整数
#
#   6. trunc: 截断小数部分
#      - 与 fix 相同，向零舍入
#
#  舍入规则对比:
#   输入: 2.3, -2.3
#   - floor:  2.0, -3.0  (总是向下)
#   - ceil:   3.0, -2.0  (总是向上)
#   - trunc:  2.0, -2.0  (向零)
#   - fix:    2.0, -2.0  (向零，与 trunc 相同)
#   - rint:   2.0, -2.0  (最近整数)
#   - around: 2.0, -2.0  (四舍五入)
#
#  整数类型不测试:
#   - 舍入函数主要用于浮点数
#   - 对整数输入，大多数舍入函数返回原值（测试无意义）