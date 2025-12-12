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

"""有理数运算函数测试

主要测试函数：
1. gcd(x1, x2) - 最大公约数
2. lcm(x1, x2) - 最小公倍数
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


# ========== GCD 测试用例 ==========

# 修改：仅测试 int32 和 int64，避开不支持的 int8/int16
@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_gcd_basic(xp, dtype):
    """基础随机测试：测试随机整数的 GCD"""
    # 固定随机种子
    numpy.random.seed(42)
    
    # 生成随机整数
    low, high = 1, 100
    np_a = numpy.random.randint(low, high, size=(3, 4)).astype(dtype)
    np_b = numpy.random.randint(low, high, size=(3, 4)).astype(dtype)
    
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.gcd(a, b)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_gcd_special_values(xp, dtype):
    """测试 GCD 的特殊数值：0, 1, 负数"""
    # gcd(a, 0) = |a|
    # gcd(0, 0) = 0
    # gcd(a, 1) = 1
    # gcd 结果总是非负的
    data_a = [0,  0, 10, -10, 1, -1]
    data_b = [0, 10,  0,  -5, 5, -5]
    
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.gcd(a, b)


# ========== LCM 测试用例 ==========

@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_lcm_basic(xp, dtype):
    """基础随机测试：测试随机整数的 LCM"""
    numpy.random.seed(42)
    
    # 注意：LCM 很容易溢出。
    # 虽然 int32/64 范围较大，但为了保险起见，输入范围仍控制较小
    low, high = 1, 20
    
    np_a = numpy.random.randint(low, high, size=(3, 4)).astype(dtype)
    np_b = numpy.random.randint(low, high, size=(3, 4)).astype(dtype)
    
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    return xp.lcm(a, b)


@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_lcm_special_values(xp, dtype):
    """测试 LCM 的特殊数值：0, 1, 负数"""
    # lcm(a, 0) = 0
    # lcm(0, 0) = 0
    # lcm(a, 1) = |a|
    # lcm 结果总是非负的
    data_a = [0,  0, 6, -6, 1, -1]
    data_b = [0, 10, 0, -4, 5, -5]
    
    a = _create_array(xp, data_a, dtype)
    b = _create_array(xp, data_b, dtype)
    return xp.lcm(a, b)


# ========== 广播测试 ==========

@testing.for_dtypes([numpy.int32, numpy.int64])
@testing.numpy_asnumpy_array_equal()
def test_rational_broadcasting(xp, dtype):
    """测试广播机制：不同形状数组的 GCD/LCM"""
    numpy.random.seed(123)
    
    # 形状 (3, 1) 和 (3,) -> 广播结果 (3, 3)
    np_a = numpy.array([[6], [12], [18]], dtype=dtype)
    np_b = numpy.array([2, 3, 4], dtype=dtype)
    
    a = _create_array(xp, np_a, dtype)
    b = _create_array(xp, np_b, dtype)
    
    # 同时测试 gcd 和 lcm
    res_gcd = xp.gcd(a, b)
    res_lcm = xp.lcm(a, b)
    
    # 返回相加结果（作为一种简单的哈希验证）
    # 使用 xp.add 确保在对应后端执行加法
    return xp.add(res_gcd, res_lcm)