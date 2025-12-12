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

"""浮点数操作函数测试

主要测试函数：
1. signbit(x) - 如果符号位被设置（即为负），返回 True。
   注意：signbit 是区分 0.0 (False) 和 -0.0 (True) 的主要方式。
"""

import numpy
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
@testing.numpy_asnumpy_array_equal()
def test_signbit_basic(xp, dtype):
    """基础随机测试：测试正负随机数"""
    # 生成包含正数和负数的随机数组
    a = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=42, scale=10.0)
    b = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=43, scale=10.0)
    # 构造一些负数
    # 修复：使用 xp.subtract 代替 - 运算符，因为 asnumpy 可能未实现运算符重载
    data = xp.subtract(a, b)
    return xp.signbit(data)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_array_equal()
def test_signbit_special_values(xp, dtype):
    """测试特殊数值：
    - 0.0 (应为 False)
    - -0.0 (应为 True)
    - 正数
    - 负数
    - inf (应为 False)
    - -inf (应为 True)
    """
    data = [
        0.0, 
        -0.0, 
        1.5, 
        -2.5, 
        float('inf'), 
        float('-inf')
    ]
    a = _create_array(xp, data, dtype)
    return xp.signbit(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_array_equal()
def test_signbit_multidim(xp, dtype):
    """测试多维数组"""
    # 构造 2x2x2 数组，手动设置一些值以确保覆盖正负
    np_data = numpy.array([
        [[1.0, -1.0], [0.0, -0.0]],
        [[2.5, -3.5], [float('inf'), float('-inf')]]
    ], dtype=dtype)
    
    a = _create_array(xp, np_data, dtype)
        
    return xp.signbit(a)