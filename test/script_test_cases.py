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
# 测试数据集 ===============================================

# 单操作数通用测试数据
UNARY_TEST_CASES = [
    np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.int32),
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

# 双操作数测试数据
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
    (np.array([], dtype=np.float32),
     np.array([], dtype=np.float32)),  # 空数组 npuarray暂不支持
    (np.array([[1, 2], [3, 4]], dtype=np.float32),
     np.array([[0], [0]], dtype=np.float32))  # 不同形状
]

# 广播测试数据
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
    (np.array([], dtype=np.float32),
    np.array([1.0, 2.0], dtype=np.float32))  # 空数组广播
]

# 矩阵乘法和点积用测试数据
MATMUL_DOT_TEST_CASES = [
    # 标量点积 (1x1)
    (np.array([2], dtype=np.float32),
     np.array([3], dtype=np.float32)),
    
    # 向量点积 (1D x 1D)
    (np.array([1, 2,9,6], dtype=np.float32),
     np.array([4, 5,10,8], dtype=np.float32)),
    
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

# 范数测试用例 (数组, ord, axis, keepdims)
NORM_TEST_CASES = [
    (np.array([1,2,3], dtype=np.float32), None, None, False),
    (np.array([[1,2],[3,4]], dtype=np.float32), 'fro', None, False),
    (np.array([[1,2],[3,4]], dtype=np.float32), 2, (0,1), True),
    (np.array([-1,0,1], dtype=np.float32), 1, None, False),
    (np.array([[np.nan,0],[0,1]], dtype=np.float32), None, None, False)
]

# QR分解测试用例 (数组, mode)
QR_TEST_CASES = [
    (np.array([[1,2],[3,4]], dtype=np.float32), 'complete'),
    (np.array([[1,2,3],[3,4,5],[5,6,7]], dtype=np.float32), 'complete'),
    (np.array([[0,1],[1,0]], dtype=np.float32), 'complete')
]

# Einsum测试用例 (表达式, 操作数列表)
EINSUM_TEST_CASES = [
    ('i,j->ij',np.array([1,2], dtype=np.float32),np.array([3,4], dtype=np.float32)),
]

# 矩阵乘方测试用例 (数组, 幂次)
MATRIX_POWER_TEST_CASES = [
    (np.array([[1,2],[3,4]], dtype=np.float32), 2),
    (np.array([[0,1],[1,0]], dtype=np.float32), 3),
    (np.array([[1,0],[0,1]], dtype=np.float32), 0),
    (np.array([[1,1],[0,1]], dtype=np.float32), -1)
]

# Inner函数测试数据
INNER_TEST_CASES = [
    # 1D vs 1D
    (np.array([1,2,3], dtype=np.float32),
     np.array([4,5,6], dtype=np.float32)),
     
    # 1D vs 2D
    (np.array([1,2], dtype=np.float32),
     np.array([[3,4],[5,6]], dtype=np.float32)),
     
    # 2D vs 1D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([5,6], dtype=np.float32)),
     
    # 2D vs 2D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([[5,6],[7,8]], dtype=np.float32)),
     
    # 包含特殊值
    (np.array([0,np.nan,1], dtype=np.float32),
     np.array([1,0,np.inf], dtype=np.float32)),
     
    # 不同形状
    (np.array([1,2,3,4], dtype=np.float32),
     np.array([5], dtype=np.float32))
]

# Outer函数测试数据
OUTER_TEST_CASES = [
    # 1D vs 1D
    (np.array([1,2,3], dtype=np.float32),
     np.array([4,5,6], dtype=np.float32)),
     
    # 1D vs 2D
    (np.array([1,2], dtype=np.float32),
     np.array([[3,4],[5,6]], dtype=np.float32)),
     
    # 2D vs 1D
    (np.array([[1,2],[3,4]], dtype=np.float32),
     np.array([5,6], dtype=np.float32)),
     
    # 包含特殊值
    (np.array([0,np.nan], dtype=np.float32),
     np.array([1,np.inf], dtype=np.float32)),
     
    # 标量
    (np.array([1], dtype=np.float32),
     np.array([2], dtype=np.float32))
]

# 行列式和符号对数行列式专用测试数据 (方阵)
DET_SLOGDET_TEST_CASES = [
    # 2x2矩阵
    np.array([[1, 2], [3, 4]], dtype=np.float32),  
    # 3x3矩阵
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
    # 特殊值矩阵
    np.array([[np.nan, 0], [np.inf, 1]], dtype=np.float32),    
    # 奇异矩阵 (行列式=0)
    np.array([[1, 2], [2, 4]], dtype=np.float32),
]

# prod有dim参数时专用测试数据
PROD_DIM_TEST_CASES = [
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 0, False),
    (np.array([[1.5, 2.3], [3.7, 4.1]], dtype=np.float32), -1, False),
    (np.array([[5, 6], [7, 8]], dtype=np.int64), 1, False),
    (np.array([[-1, -2], [-3, -4]], dtype=np.int32), -1, True),
    (np.array([[1]], dtype=np.float32), 0, False),
    (np.array([1, 2, 3, 4], dtype=np.float32), 0, True),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32), 2, False),
    (np.array([[0, 0], [0, 0]], dtype=np.float32), 1, False),
    (np.array([[np.nan, 2], [3, np.inf]], dtype=np.float32), 0, True),
    (np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float64), 1, False),
    (np.array([[1, 2], [3, 4]], dtype=np.float32), -2, True),
]

# nanprod有dim参数时专用测试数据
NANPROD_DIM_TEST_CASES = [
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 0, False),
    (np.array([[np.nan, 2], [3, np.inf]], dtype=np.float32), 0, True),
    (np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32), 1, False),
    (np.array([[-np.inf, np.nan], [np.nan, np.inf]], dtype=np.float32), 0, True),
    (np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]], dtype=np.float32), 1, False),
    (np.array([[1.5, np.nan], [3.7, 4.2]], dtype=np.float32), 1, True)
]

# sum有dim参数时专用测试数据
SUM_DIM_TEST_CASES = [
    (np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32), 0, True),
    (np.array([[[1.0, 2.0], [3.0, -4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32), 2, True),
    (np.array([1.0, -2.0, 3.0, 4.0], dtype=np.float32), 0, True),
    (np.random.rand(10, 20, 5).astype(np.float32), 2, True),
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), -1, False),
    (np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32), 0, True),
    (np.ones((2, 3, 4, 5, 6), dtype=np.float32), 3, True),
    (np.array([[1.5, 2.7], [3.1, 4.9]], dtype=np.float32), 1, True),
]

# nansum有dim参数时专用测试数据
NANSUM_DIM_TEST_CASES = [
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 0, False),
    (np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32), 0, True),  
    (np.array([[np.nan, np.inf], [-np.inf, np.nan]], dtype=np.float32), 1, False),
    (np.array([[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float32), 0, True),
    (np.array([[[[1.0, np.nan], [np.nan, 4.0]], [[5.0, 6.0], [7.0, np.nan]]]], dtype=np.float32), 3, False),
]

#cumprod和cumsum测试用例
CUM_TEST_CASES = [
    # 2D数组测试用例
    (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 0),
    (np.array([[5.0, -2.0], [0.0, 8.0]], dtype=np.float32), 1),
    (np.array([[1.5, 2.7], [-3.2, 4.9]], dtype=np.float32), 0),
    # 3x2数组
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32), 1),
    # 包含负数和零
    (np.array([[-1.0, 0.0], [2.5, -3.7]], dtype=np.float32), 0),
    # 小数数组
    (np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32), 1),
    # 3D数组测试用例
    (np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32), 0),
    (np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32), 2),
    # 2x3x2数组
    (np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32), 1),
    # 1D数组测试用例
    (np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), 0),
    (np.array([-1.0, 0.0, 2.5, -3.7], dtype=np.float32), 0),  
    # 带inf的数组
    (np.array([[1.0, np.inf], [3.4, 7.6]], dtype=np.float32), 0),

    # NaN数组测试
    (np.full((3, 3), np.nan, dtype=np.float32), 1),
    (np.array([[1.0, np.nan, np.inf], [0.0, -np.inf, 2.5]], dtype=np.float32), 1),
    (np.array([[[1.0, np.nan], [np.inf, 2.0]],[[-1.0, 0.0], [np.nan, -np.inf]]], dtype=np.float32), 2),
    (np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32), 0),
]

#cross测试用例
CROSS_TEST_CASES = [
    # 1D 向量测试用例
    (np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([4.0, 5.0, 6.0], dtype=np.float32), 0),     
    (np.array([-1.0, 0.0, 1.0], dtype=np.float32),
     np.array([2.0, -2.0, 0.0], dtype=np.float32), 0),
    # 2D 数组测试用例 - 多个向量
    (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
     np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32), 1),
    (np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
     np.array([[4.0], [5.0], [6.0]], dtype=np.float32), 0),
    # 测试不同轴
    (np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32),
     np.array([[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32), 2),
    (np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
     np.array([[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]], dtype=np.float32), 1),
    # 3D 数组测试用例
    (np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
               [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]], dtype=np.float32),
     np.array([[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
               [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]]], dtype=np.float32), 2), 
    # 正交向量测试
    (np.array([1.0, 0.0, 0.0], dtype=np.float32),
     np.array([0.0, 1.0, 0.0], dtype=np.float32), 0),
    # 平行向量测试（叉积为零）
    (np.array([-1.0, -2.0, -3.0], dtype=np.float32),
     np.array([0.5, 1.0, 1.5], dtype=np.float32), 0),
    # 零向量测试
    (np.array([0.0, 0.0, 0.0], dtype=np.float32),
     np.array([1.0, 2.0, 3.0], dtype=np.float32), 0),
]

# 复数测试专用数据
COMPLEX_UNARY_TEST_CASES = [
    np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float32),
    np.array([-1.5, 2.7, -0.3, 4.0], dtype=np.float32),
    np.array([-10], dtype=np.float32),
    np.array([[1, -2], [-3, 4]], dtype=np.float32),
    np.array([-10, 0, 10], dtype=np.float32),
    np.array([[-1, 0], [0, 1]], dtype=np.float32),
    np.array([-3-1j, -2+2j, -1-3j, 0+0j, 1+1j, 2-2j, 3+3j], dtype=np.complex64),
    np.array([-1.5+0.5j, 2.7-1.2j, -0.3+2.1j, 4.0-0.8j], dtype=np.complex64),
    np.array([-10+5j], dtype=np.complex64),
    np.array([[1+2j, -2-1j], [-3+0j, 4-3j]], dtype=np.complex64),
    np.array([-10-10j, 0+0j, 10+10j], dtype=np.complex64),
    np.array([[-1+1j, 0-2j], [0+3j, 1-1j]], dtype=np.complex64)
]

# log类函数测试数据
LOG_UNARY_TEST_CASES = [
    np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float32),
    np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32),
    np.array([0.001, 0.01, 0.1], dtype=np.float32),
    np.array([2.71828], dtype=np.float32),  # e，log(e)=1
    np.array([[1.0, 4.0], [9.0, 16.0]], dtype=np.float32),  # 平方数
    np.array([np.e**-2, np.e**-1, np.e**0, np.e**1, np.e**2], dtype=np.float32),
    np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32),
    np.array([[2.0, 4.0], [8.0, 16.0]], dtype=np.float32)
]

GCD_TEST_CASES = [
    # 基础正整数测试
    (np.array([4, 6, 8], dtype=np.int32), np.array([6, 9, 12])),   # 默认 int32
    (np.array([15, 25, 35], dtype=np.int64), np.array([20, 30, 40], dtype=np.int64)),  # 指定 int64
    
    # 包含1的场景
    (np.array([1, 1, 1], dtype=np.int32), np.array([5, 10, 15])),
    (np.array([7, 11, 13]), np.array([1, 1, 1], dtype=np.int32)),  # 输出指定
    
    # 包含0的场景
    (np.array([0, 0, 0], dtype=np.int32), np.array([5, -7, 9])),
    (np.array([-3, 6, -9]), np.array([0, 0, 0], dtype=np.int32)),
    (np.array([0, 5, -3]), np.array([0, 0, 6])),  # gcd(0,0) 未定义
    
    # 负数场景
    (np.array([-4, -6, -8], dtype=np.int32), np.array([6, -9, 12])),
    (np.array([-10, 15, -20], dtype=np.int64), np.array([-15, -25, 30], dtype=np.int64)),  # 指定 int64
    
    # 相同数字场景
    (np.array([5, -7, 0]), np.array([5, -7, 0], dtype=np.int32)),
    
    # 多维数组
    (np.array([[4, 6], [8, 10]], dtype=np.int32), np.array([[6, 9], [12, 15]], dtype=np.int32)),
    (np.array([[[12, 18], [24, 30]]], dtype=np.int32), np.array([[[18, 24], [30, 36]]], dtype=np.int32)),
    
    # 大整数
    (np.array([123456789, 987654321], dtype=np.int64), np.array([135792468, 864209753], dtype=np.int64)),
    
    # 广播机制
    (np.array([2, 3, 4], dtype=np.int32), np.array([[6], [9], [12]], dtype=np.int32))
]

LCM_TEST_CASES = [
    # 基础正整数测试
    (np.array([4, 6, 8], dtype=np.int32), np.array([6, 9, 12])),
    (np.array([15, 25, 35], dtype=np.int64), np.array([20, 30, 40], dtype=np.int64)),
    
    # 包含1的场景
    (np.array([1, 1, 1], dtype=np.int32), np.array([5, 10, 15])),
    (np.array([7, 11, 13]), np.array([1, 1, 1], dtype=np.int32)),
    
    # 包含0的场景
    (np.array([0, 5, -3], dtype=np.int32), np.array([5, 0, 6], dtype=np.int32)),
    
    # 负数场景
    (np.array([-4, -6, -8], dtype=np.int32), np.array([6, -9, 12])),
    (np.array([-10, 15, -20], dtype=np.int64), np.array([-15, -25, 30], dtype=np.int64)),  # 指定 int64
    
    # 相同数字场景
    (np.array([5, -7, 9]), np.array([5, -7, 9], dtype=np.int32)),
    
    # 互质场景
    (np.array([3, 5, 7], dtype=np.int32), np.array([4, 6, 8], dtype=np.int32)),
    
    # 多维数组
    (np.array([[4, 6], [8, 10]], dtype=np.int32), np.array([[6, 9], [12, 15]], dtype=np.int32)),
    (np.array([[[12, 18], [24, 30]]], dtype=np.int32), np.array([[[18, 24], [30, 36]]], dtype=np.int32)),
    
    # 大整数
    (np.array([123456789, 987654321], dtype=np.int64), np.array([135792468, 864209753], dtype=np.int64)),
    
    # 广播机制
    (np.array([2, 3, 4], dtype=np.int32), np.array([[6], [9], [12]], dtype=np.int32))
]

AROUND_TEST_CASES = [
    # float32 正常值
    (np.array([-1.5, 2.7, -0.3, 4.0], dtype=np.float32), 0),
    (np.array([-10], dtype=np.float32), 1),
    (np.array([[1, -2], [-3, 4]], dtype=np.float32), 2),
    (np.array([1.234567, -9.876543], dtype=np.float32), 3),

    # float32 特殊值
    (np.array([np.inf, -np.inf, np.nan], dtype=np.float32), 2),
    (np.array([0.0, -0.0, 0.0], dtype=np.float32), 0),   # 正负零混合
    (np.array([1e38, -1e38, 1e-38], dtype=np.float32), 5),  # 极大极小值
    (np.array([2.2250739e-308], dtype=np.float32), 6),  # 最小正规格化数

    # float32 重复值
    (np.array([1.0, 1.0, 1.0], dtype=np.float32), 0),
    (np.array([-1.0, -1.0, -1.0], dtype=np.float32), 0),

    # float32 矩阵 + 特殊值
    (np.array([[np.nan, 0], [0, np.inf]], dtype=np.float32), 1),
    (np.array([[3.14159, -2.71828], [1.41421, -0.57721]], dtype=np.float32), 4),
]

FIX_TEST_CASES = [
    np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float32),
    np.array([2.9, -2.9, 123.0, -123.0], dtype=np.float32),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([[1.9, -1.9], [2.5, -2.5]], dtype=np.float32),
    np.array([[[0.1], [-0.1]]], dtype=np.float32)
]

TRUNC_TEST_CASES = [
    np.array([-3.9, -2.1, -0.9, 0.9, 2.1, 3.9], dtype=np.float32),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([0.0, -0.0], dtype=np.float32),
    np.array([[10.5, -10.5], [20.1, -20.1]], dtype=np.float32),
    np.array([[[30.7], [-30.7]]], dtype=np.float32)
]

RINT_TEST_CASES = [
    # float32
    np.array([-1.5, -0.5, 0.5, 1.5]),
    np.array([2.3, 2.7, -2.3, -2.7], dtype=np.float32),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([[1.2, 2.5], [-3.7, -4.5]], dtype=np.float32),
    np.array([[[1.1, -1.1]]], dtype=np.float32),
    # float64
    np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64),
    np.array([123456.789, -98765.4321], dtype=np.float64),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
    np.array([[10.2, -20.5], [30.7, -40.9]], dtype=np.float64)
]

FLOOR_TEST_CASES = [
    # float32
    np.array([-1.7, -0.2, 0.2, 1.7]),
    np.array([2.9, -2.9, 123.0, -123.0], dtype=np.float32),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([[3.3, -3.3], [4.9, -4.9]], dtype=np.float32),
    # float64
    np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float64),
    np.array([2.9, -2.9, 1e308, -1e308], dtype=np.float64),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
    np.array([[123.456, -654.321]], dtype=np.float64)
]

CEIL_TEST_CASES = [
    # float32
    np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float32),
    np.array([2.9, -2.9, 123.0, -123.0], dtype=np.float32),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
    np.array([[3.3, -3.3], [4.9, -4.9]], dtype=np.float32),
    # float64
    np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float64),
    np.array([2.9, -2.9, 1e308, -1e308], dtype=np.float64),
    np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
    np.array([[123.456, -654.321]], dtype=np.float64)
]

CLIP_TEST_CASES = [
    # float32 基础测试
    (
        np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32),  # a
        np.array([-1.0], dtype=np.float32),                      # a_min
        np.array([1.0], dtype=np.float32)                        # a_max
    ),
    (
        np.array([10.0, -10.0, 5.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([6.0], dtype=np.float32)
    ),
    (
        np.array([np.inf, -np.inf, np.nan, 3.3], dtype=np.float32),
        np.array([-5.0], dtype=np.float32),
        np.array([5.0], dtype=np.float32)
    ),
    (
        np.array([[3.3, -3.3], [4.9, -4.9]], dtype=np.float32),
        np.array([-4.0], dtype=np.float32),
        np.array([4.0], dtype=np.float32)
    ),

    # float64 边界测试
    (
        np.array([-1.7, -0.2, 0.2, 1.7], dtype=np.float64),
        np.array([-1.0], dtype=np.float64),
        np.array([1.0], dtype=np.float64)
    ),
    (
        np.array([2.9, -2.9, 1e308, -1e308], dtype=np.float64),
        np.array([-10.0], dtype=np.float64),
        np.array([10.0], dtype=np.float64)
    ),
    (
        np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
        np.array([-1e5], dtype=np.float64),
        np.array([1e5], dtype=np.float64)
    ),
    (
        np.array([[123.456, -654.321]], dtype=np.float64),
        np.array([-500.0], dtype=np.float64),
        np.array([500.0], dtype=np.float64)
    ),
]

CLIP_OBJ_OBJ_CASES = [
    # float32
    (
        np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32),
        -1.0,   # a_min
        1.0     # a_max
    ),
    (
        np.array([10.0, -10.0, 5.0], dtype=np.float32),
        0.0,
        6.0
    ),
    (
        np.array([np.inf, -np.inf, np.nan, 3.3], dtype=np.float32),
        -5.0,
        5.0
    ),

    # float64
    (
        np.array([2.9, -2.9, 1e308, -1e308], dtype=np.float64),
        -10.0,
        10.0
    ),
    (
        np.array([np.inf, -np.inf, np.nan], dtype=np.float64),
        -1e5,
        1e5
    ),
]

CLIP_OBJ_ARR_CASES = [
    (
        np.array([1.0, 5.0, 10.0], dtype=np.float32),
        0.0,  # a_min
        np.array([2.0, 6.0, 8.0], dtype=np.float32)  # element-wise upper bound
    ),
    (
        np.array([-100.0, 0.0, 100.0], dtype=np.float64),
        -50.0,
        np.array([0.0, 10.0, 50.0], dtype=np.float64)
    ),
]

CLIP_ARR_OBJ_CASES = [
    (
        np.array([1.0, 5.0, 10.0], dtype=np.float32),
        np.array([0.0, 3.0, 8.0], dtype=np.float32),  # element-wise lower bound
        9.0  # a_max
    ),
    (
        np.array([-100.0, 0.0, 100.0], dtype=np.float64),
        np.array([-200.0, -1.0, 50.0], dtype=np.float64),
        80.0
    ),
]

NAN_TO_NUM_TEST_CASES = [
    # 1. 基础场景（完整参数传递）
    (np.array([np.nan, np.inf, -np.inf, 3.14, -2.718], dtype=np.float32),  # 转换为NPUArray
     0.0, 1e5, -1e5),
    
    # 2
    (np.array([0.0, -0.0, np.nan, np.inf], dtype=np.float32),
     10.0, None, None), 
    
    # 3
    (np.array([[np.nan, 100.0], [-np.inf, np.inf]], dtype=np.float32),
     -5.0, 100.0, -100.0),
]

MAX_MIN_TEST_CASES = [
    # 基础float32测试 - 相同形状
    (np.array([1.5, 2.7, -0.3], dtype=np.float32), 
     np.array([2.0, 1.5, -0.1], dtype=np.float32)),
    
    # 广播测试 - 标量与数组
    (np.array([-1.5, 2.7, -0.3], dtype=np.float32), 
     np.array(0.0, dtype=np.float32)),
    
    # 广播测试 - 不同维度
    (np.array([[1, 2], [3, 4]], dtype=np.float32), 
     np.array([5, 6], dtype=np.float32)),
    
    # float64类型测试
    (np.array([1.234567890123, -9.876543210987], dtype=np.float64), 
     np.array([2.345678901234, -8.765432109876], dtype=np.float64)),
    
    # 特殊值测试 - inf和-inf
    (np.array([np.inf, -np.inf, 5.0], dtype=np.float32), 
     np.array([10.0, np.inf, -np.inf], dtype=np.float32)),
    
    # 特殊值测试 - nan处理
    (np.array([np.nan, 0.0, np.nan], dtype=np.float32), 
     np.array([5.0, np.nan, np.nan], dtype=np.float32)),
    
    # 正负零测试
    (np.array([0.0, -0.0, 0.0], dtype=np.float32), 
     np.array([-0.0, 0.0, -0.0], dtype=np.float32)),
    
    # 极值测试
    (np.array([1e38, -1e38, 1e-38], dtype=np.float32), 
     np.array([2e38, -2e38, 2e-38], dtype=np.float32)),
    
    # 整数与浮点数混合
    (np.array([1, 2, 3], dtype=np.int32), 
     np.array([2.5, 1.5, 3.5], dtype=np.float32)),
    
    # 重复值测试
    (np.array([5.0, 5.0, 5.0], dtype=np.float32), 
     np.array([5.0, 5.0, 5.0], dtype=np.float32)),
    
    # 高维数组测试
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32), 
     np.array([[[8, 7], [6, 5]], [[4, 3], [2, 1]]], dtype=np.float32)),
    
    # 使用默认dtype测试
    (np.array([-3.2, 4.7, -1.8], dtype=np.float32), 
     np.array([-2.1, 3.9, -0.5], dtype=np.float32)),
]

SORT_TEST_CASES = [
    # 一维数组，沿轴0排序，不保持原顺序
    (np.array([-5, 0, -2, 7, -1], dtype=np.float32), 0, False),

    # 二维数组，沿轴0排序（按行排序）
    (np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]], dtype=np.float32), 0, False),

    # 二维数组，沿轴1排序（按列排序）
    (np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]], dtype=np.float32), 1, False),

    # 三维数组，沿最后一个轴排序
    (np.array([[[3, 1], [4, 2]], [[5, 8], [9, 6]]], dtype=np.float32), -1, False),

    # 全为相同元素的数组
    (np.array([2, 2, 2, 2, 2], dtype=np.float32), 0, False),

    # 空数组边界测试
    (np.array([], dtype=np.float32), 0, False),

    # 单个元素数组
    (np.array([42], dtype=np.float32), 0, False),

    # 包含NaN值的数组
    (np.array([3.5, np.nan, 1.2, np.nan, 7.8], dtype=np.float32), 0, False),

    # 大数值范围测试
    (np.array([1e10, -1e10, 1e-10, -1e-10], dtype=np.float32), 0, False),
]

LDEXP_TEST_CASES = [
    # === 基础测试 ===
    (np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([1, 2, 3], dtype=np.int32)),

    (np.array([-1.5, 0.0, 2.5], dtype=np.float32),
     np.array([-1, 0, 1], dtype=np.int32)),

    # === 广播测试 ===
    (np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32),
     np.array([1], dtype=np.int32)),  # scalar-like broadcast

    (np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([[0], [1]], dtype=np.int32)),  # 3 vs (2,1)

    # === 特殊数值 ===
    (np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
     np.array([1, -1, 0], dtype=np.int32)),

    (np.array([0.0, -0.0, 1.0], dtype=np.float32),
     np.array([10, -10, 0], dtype=np.int32)),  # 正负零处理

    # === 极大极小值 ===
    (np.array([1e-38, 1e-20, 1e20], dtype=np.float32),
     np.array([10, -10, 1], dtype=np.int32)),

    # === 高维张量 ===
    (np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32),
     np.array([1], dtype=np.int32)),

    # === 指数 float32（numpy 允许）===
    (np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([1.0, -1.0, 0.0], dtype=np.float32)),
]

COPYSIGN_TEST_CASES = [
    # === 基础测试 ===
    (np.array([1.0, -2.0, 3.0], dtype=np.float32),
     np.array([-1.0, 2.0, -3.0], dtype=np.float32)),

    (np.array([-1.5, 0.0, 2.5], dtype=np.float32),
     np.array([1.0, -1.0, 0.0], dtype=np.float32)),

    # === 广播测试 ===
    (np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32),
     np.array([1.0], dtype=np.float32)),  # 1 → (2,2)

    (np.array([1.0, 2.0, 3.0], dtype=np.float32),
     np.array([[0.0], [1.0]], dtype=np.float32)),  # 3 vs (2,1)

    # === 特殊数值 ===
    (np.array([np.inf, -np.inf, np.nan], dtype=np.float32),
     np.array([-1.0, np.nan, 1.0], dtype=np.float32)),

    (np.array([0.0, -0.0, 1.0], dtype=np.float32),
     np.array([-0.0, +0.0, -1.0], dtype=np.float32)),  # 处理 ±0 的符号

    # === 极大极小 ===
    (np.array([1e38, -1e-38, 1e-38], dtype=np.float32),
     np.array([-1e38, 1e38, -1e-38], dtype=np.float32)),

    # === 多维测试 ===
    (np.array([[1, -2], [-3, 4]], dtype=np.float32),
     np.array([[0, 1], [1, 0]], dtype=np.float32)),

    # === NaN 与广播混合 ===
    (np.array([[np.nan], [0], [1]], dtype=np.float32),
     np.array([0.5, np.inf, -1.0], dtype=np.float32)),
]

# 在script_test_cases.py中添加
DEGREES_TEST_CASES = [
    np.array([0]),
    np.array([np.pi]),
    np.array([np.pi / 2]),
    np.array([np.pi / 3]),
    np.array([2 * np.pi]),
    np.array([-np.pi]),
    np.array([np.pi / 4]),
]
