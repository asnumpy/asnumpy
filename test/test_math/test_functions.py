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

from test_data import UNARY_TEST_CASES, BINARY_TEST_CASES, BROADCAST_TEST_CASES, MATMUL_DOT_TEST_CASES

import numpy as np
import asnumpy as ap

# 函数注册表 =================================================

# 单操作数函数注册表 (函数名, numpy函数, asnumpy函数, 测试用例列表)
# 使用共享数据集UNARY_TEST_CASES，可额外添加自定义用例
UNARY_FUNCTIONS = [
    (
        "absolute",
        np.absolute,
        ap.absolute,
        UNARY_TEST_CASES
    ),
    (
        "sign",
        np.sign,
        ap.sign,
        UNARY_TEST_CASES
    ),
    (
        "fabs",
        np.fabs,
        ap.fabs,
        UNARY_TEST_CASES
    ),
        (
        "relu",
        lambda x: np.maximum(x, 0),  # NumPy 没有内置 relu，用 maximum 模拟
        ap.relu,
        UNARY_TEST_CASES
    ),
        (
        "gelu",
        lambda x: x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3)))),  # GELU 近似公式
        ap.gelu,
        UNARY_TEST_CASES
    ),
]

# 双操作数函数注册表 (函数名, numpy函数, asnumpy函数, 测试用例列表)
# 基础用例BINARY_TEST_CASES + 广播用例BROADCAST_TEST_CASES
BINARY_FUNCTIONS = [
    (
        "heaviside",
        np.heaviside,
        ap.heaviside,
        BINARY_TEST_CASES + BROADCAST_TEST_CASES
    ),
    (
        "matmul",
        np.matmul,
        ap.matmul,
        MATMUL_DOT_TEST_CASES
    ),
    (
        "dot",
        np.dot,
        ap.dot,
        MATMUL_DOT_TEST_CASES
    )
]
