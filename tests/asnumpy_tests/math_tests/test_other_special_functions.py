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

针对已记录的算子限制进行标注：
1. sinc: 基础功能支持 float32, float64。
2. float16: 触发 RuntimeError (Unsupported py::dtype for aclDataType)，C++ 映射缺失。
3. int32: 触发 RuntimeError 161002，底层算子输出类型不支持。
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
    return xp.ndarray.from_numpy(np_arr)


# ========== 1. Sinc 正常链路 (Float32/64) ==========


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose(atol=1e-5, rtol=1e-5)
def test_sinc_basic(xp, dtype):
    """测试 sinc 基础功能（已知支持的浮点类型）"""
    data = [-3.0, -1.5, 0.5, 2.0, 3.5]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_dtypes([numpy.float64])
@testing.numpy_asnumpy_allclose()
def test_sinc_zero(xp, dtype):
    """测试 sinc(0) = 1"""
    data = [0.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


# ========== 2. 异常与 Bug 记录 (XFAIL) ==========


@pytest.mark.xfail(reason="Bug: C++ core missing mapping from float16 to aclDataType (Unsupported py::dtype)")
@testing.for_dtypes([numpy.float16])
def test_sinc_float16_mapping_xfail(xp, dtype):
    """
    记录：sinc 虽然在硬件层面支持 float16，但 asnumpy 绑定层尚未处理该映射。
    """
    a = _create_array(xp, [0.5], dtype)
    return xp.sinc(a)


@pytest.mark.xfail(reason="Bug: aclnnSinc does not support INT32 output (RuntimeError 161002)")
@testing.for_dtypes([numpy.int32])
def test_sinc_int_output_xfail(xp, dtype):
    """
    记录：输入 int32 时，由于输出也被设为 int32，导致 aclnn 拒绝执行。
    """
    data = [1, 2]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)
