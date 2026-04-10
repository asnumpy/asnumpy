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

import numpy as np
# 共享测试数据集 ===============================================

# 单操作数通用测试数据
UNARY_TEST_CASES = [
    np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
    np.array([-1.5, 2.7, -0.3, 4.0], dtype=np.float32),
    np.array([-10], dtype=np.float32),
    np.array([[1, -2], [-3, 4]], dtype=np.float32),
    np.array([-10, 0, 10], dtype=np.float32),
    np.array([[-1, 0], [0, 1]], dtype=np.float32),
    # 新增边缘测试用例
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([0.0, -0.0, 0.0], dtype=np.float32),  # 正负零混合
    # np.array([], dtype=np.float32),  # 空数组
    np.array([1e38, -1e38, 1e-38], dtype=np.float32),  # 极大极小值
    np.array([1.0, 1.0, 1.0], dtype=np.float32),  # 全正值
    np.array([-1.0, -1.0, -1.0], dtype=np.float32),  # 全负值
    np.array([[np.nan, 0], [0, np.inf]], dtype=np.float32),  # 特殊值矩阵
    np.array([2.2250738585072014e-308], dtype=np.float32)  # 最小正规格化浮点数
]

# 双操作数通用测试数据
BINARY_TEST_CASES = [
    (np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
     np.array([0.5], dtype=np.float32)),
    (np.array([-1.5, 0.0, 2.7], dtype=np.float32),
     np.array([1.0], dtype=np.float32)),
    (np.array([0, 0, 0], dtype=np.float32),
     np.array([0.5], dtype=np.float32)),
    (np.array([1, 2, 3], dtype=np.float32),
     np.array([0.0], dtype=np.float32)),
    (np.array([-1, -2, -3], dtype=np.float32),
     np.array([1.0], dtype=np.float32)),
    (np.array([[0, 1], [-1, 0]], dtype=np.float32),
     np.array([[0.5, 0.5]], dtype=np.float32)),
    # 新增边缘测试用例
    (np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
     np.array([1.0, np.nan, 0.0], dtype=np.float32)),
    (np.array([0.0, 0.0, 0.0], dtype=np.float32),
     np.array([0.0, 1.0, -1.0], dtype=np.float32)),  # 零值处理
    (np.array([1e20, -1e20, 0], dtype=np.float32),
     np.array([1e-20, 1e20, 0], dtype=np.float32)),  # 大数和小数
    # (np.array([], dtype=np.float32),
    #  np.array([], dtype=np.float32)),  # 空数组 npuarray暂不支持
    (np.array([[1, 2], [3, 4]], dtype=np.float32),
     np.array([[0], [0]], dtype=np.float32))  # 不同形状
]

# 广播测试专用数据
BROADCAST_TEST_CASES = [
    (np.array([[-1, 0, 1], [2, -2, 0]], dtype=np.float32),
     np.array([0.5, 1.0, 0.0], dtype=np.float32)),
    (np.array([1, 2, 3], dtype=np.float32),
     np.array([[0.5], [1.0]], dtype=np.float32)),
    (np.array([0], dtype=np.float32),
     np.array([0.1, 0.2, 0.3], dtype=np.float32)),
    # 新增边缘广播用例
    (np.array([[np.nan], [0], [1]], dtype=np.float32),
     np.array([0.5, np.inf, -1.0], dtype=np.float32)),
    (np.array([1e38, -1e38], dtype=np.float32),
     np.array([[0.0], [1.0]], dtype=np.float32)),
    # (np.array([], dtype=np.float32),
    # np.array([1.0, 2.0], dtype=np.float32))  # 空数组广播
]

# 矩阵乘法和点积专用测试数据
MATMUL_DOT_TEST_CASES = [
    # 标量点积 (1x1)
    (np.array([2], dtype=np.float32),
     np.array([3], dtype=np.float32)),
    
    # 向量点积 (1D x 1D)
    (np.array([1, 2, 3], dtype=np.float32),
     np.array([4, 5, 6], dtype=np.float32)),
    
    # 矩阵乘法 (2D x 2D)
    (np.array([[1, 2], [3, 4]], dtype=np.float32),
     np.array([[5, 6], [7, 8]], dtype=np.float32)),
    
    # 高维张量乘法 (3D x 3D)
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32),
     np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)),
    
    # 边界情况：包含特殊值
    (np.array([[np.inf, 0], [np.nan, 1e38]], dtype=np.float32),
     np.array([[0, 1], [1, 0]], dtype=np.float32))
]
