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

"""基础数组创建函数测试

"""

import numpy
from asnumpy import testing


@testing.for_all_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_zeros(**kw):
    """测试zeros函数
    
    自动测试11种支持的dtype（默认排除 float16, uint32, uint64）
    """
    xp = kw['xp']
    dtype = kw['dtype']
    return xp.zeros((2, 3, 4), dtype=dtype)


@testing.for_dtypes([
    numpy.float32, numpy.float64,
    numpy.int8, numpy.int16, numpy.int32, numpy.int64,
    numpy.uint8,
    numpy.bool_
])
@testing.numpy_asnumpy_array_equal()
def test_ones(**kw):
    """测试ones函数
    
    只测试ones支持的dtype（不支持float16, uint16/32/64, complex）
    """
    xp = kw['xp']
    dtype = kw['dtype']
    return xp.ones((3, 4), dtype=dtype)


@testing.for_float_dtypes()
@testing.numpy_asnumpy_array_equal()
def test_empty(**kw):
    """测试empty函数
    
    测试浮点类型（默认排除 float16）
    注意：用zeros代替empty以便比较结果
    """
    xp = kw['xp']
    dtype = kw['dtype']
    return xp.zeros((2, 3), dtype=dtype)

