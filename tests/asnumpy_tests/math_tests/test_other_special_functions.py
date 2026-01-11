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

"""其他特殊数学函数测试

主要测试函数：
1. sinc(x) - 计算 sin(pi*x)/(pi*x)
"""

import numpy
import pytest
from asnumpy import testing

# ========== 辅助函数 ==========

def _create_array(xp, data, dtype):
    """辅助函数：创建数组
    
    解决 asnumpy 尚未实现 xp.array() 接口的问题。
    """
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ========== 测试用例 ==========

@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_basic(xp, dtype):
    """基础随机测试：测试常规范围内的浮点数"""
    # 修复：必须在函数内部设置固定种子
    # 否则 xp=numpy 和 xp=asnumpy 会生成两组不同的随机数，导致对比失败
    numpy.random.seed(42) 
    
    # 生成 -5 到 5 之间的随机数
    np_a = numpy.random.uniform(low=-5.0, high=5.0, size=(10, 10)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_sinc_at_zero(xp, dtype):
    """测试 sinc 在 x=0 处的行为"""
    data = [0.0, -0.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_integers(xp, dtype):
    """测试整数点的值"""
    data = [-5.0, -3.0, -1.0, 1.0, 2.0, 4.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_halves(xp, dtype):
    """测试半整数点"""
    data = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_special_values(xp, dtype):
    """测试特殊数值：无穷大和 NaN"""
    data = [float('inf'), float('-inf'), float('nan')]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_multidim(xp, dtype):
    """测试多维数组"""
    # 修复：设置随机种子，确保两轮执行数据一致
    numpy.random.seed(123)
    
    np_data = numpy.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sinc(a)