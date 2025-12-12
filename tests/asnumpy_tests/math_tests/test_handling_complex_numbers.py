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

"""复数处理函数测试

主要测试函数：
1. real(x) - 返回复数的实部
"""

import numpy
from asnumpy import testing

# ========== 辅助函数 ==========

def _create_array(xp, data, dtype):
    """辅助函数：创建数组
    
    由于 asnumpy 尚未实现 xp.array() 接口，
    这里统一封装创建逻辑：如果是 asnumpy 环境，则通过 from_numpy 转换。
    """
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ========== 测试用例 (仅测试 float32 和 complex64) ==========

# 使用 for_dtypes 明确指定仅测试 float32 和 complex64
# 避开 float64, complex128 (报错 DT_DOUBLE/DT_COMPLEX128) 和 整数 (报错不支持)
@testing.for_dtypes([numpy.float32, numpy.complex64])
@testing.numpy_asnumpy_array_equal()
def test_real_basic(xp, dtype):
    """基础随机测试：覆盖常规形状 (3, 4)"""
    a = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=42)
    return xp.real(a)

@testing.for_dtypes([numpy.complex64]) # 仅测试 complex64
@testing.numpy_asnumpy_array_equal()
def test_real_complex_special_values(xp, dtype):
    """测试复数类型的特定数值"""
    data = [
        1+2j,    # 普通复数
        -2.5+3j, # 实部为负
        0+5j,    # 纯虚数 (实部为0)
        3-4j,    # 虚部为负
        -1-1j,   # 实虚皆负
        0+0j,    # 零
        5+0j     # 纯实数 (虚部为0)
    ]
    a = _create_array(xp, data, dtype)
    return xp.real(a)

@testing.for_dtypes([numpy.float32]) # 仅测试 float32
@testing.numpy_asnumpy_array_equal()
def test_real_float_special_values(xp, dtype):
    """测试浮点类型的特定数值"""
    data = [0.0, -0.0, 1.5, -2.5, 100.0, -100.0]
    a = _create_array(xp, data, dtype)
    return xp.real(a)

@testing.for_dtypes([numpy.float32, numpy.complex64])
@testing.numpy_asnumpy_array_equal()
def test_real_multidim(xp, dtype):
    """测试多维数组形状：3D 数组"""
    a = testing.shaped_random((2, 2, 2), xp=xp, dtype=dtype, seed=42)
    return xp.real(a)

@testing.for_dtypes([numpy.float32, numpy.complex64])
@testing.numpy_asnumpy_array_equal()
def test_real_2d_explicit(xp, dtype):
    """测试显式构造的 2D 数组"""
    if numpy.dtype(dtype).kind == 'c':
            # 2x2 复数矩阵
            val = [[1+1j, 2+0j], [0+3j, -1-1j]]
    else:
            # 2x2 实数矩阵
            val = [[1, 2], [0, -1]]
            
    a = _create_array(xp, val, dtype)
    return xp.real(a)

@testing.for_dtypes([numpy.float32]) # 仅测试 float32
@testing.numpy_asnumpy_array_equal()
def test_real_non_complex_identity(xp, dtype):
    """专门测试非复数类型：
    验证对实数调用 real() 是否正确返回原数组（数值不变）
    """
    a = testing.shaped_random((3, 4), xp=xp, dtype=dtype, seed=42)
    return xp.real(a)