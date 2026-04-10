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

"""测试辅助函数

提供便捷的测试数据生成和处理工具。
"""
__all__ = [
    'shaped_arange',
    'shaped_random',
    'shaped_reverse_arange',
    'assert_array_list_equal',
    'suppress_warnings',
    'with_seed',
    'generate_test_data',
    'TEST_SHAPES',
    'TEST_DTYPES',
    'TEST_ORDERS',
]

import functools
import numpy


def shaped_arange(shape, dtype=numpy.float64, order='C', xp=None, start=0):
    """生成指定形状的序列数组
    
    生成从start开始的连续整数序列，然后reshape成指定形状。
    这在测试中非常有用，因为可以轻松验证数组操作的正确性。
    
    Args:
        start: 起始值，默认为0。用于生成从指定值开始的序列。
    """
    if xp is None:
        xp = numpy
    
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)
    
    # 计算总元素数
    size = 1
    for dim in shape:
        size *= dim
    
    # 生成序列 (从 start 到 start+size)
    if xp is numpy:
        arr = numpy.arange(start, start + size, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # 对于asnumpy，先用numpy生成再转换
        arr = numpy.arange(start, start + size, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def shaped_random(shape, dtype=numpy.float64, scale=1.0, seed=None, xp=None):
    """生成指定形状的随机数组
    
    生成服从均匀分布的随机数数组。
    """
    if xp is None:
        xp = numpy
    
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)
    
    # 设置随机种子
    if seed is not None:
        numpy.random.seed(seed)
    
    # 生成随机数
    if xp is numpy:
        arr = numpy.random.random(shape).astype(dtype)
        return arr * scale
    else:
        # 对于asnumpy，先用numpy生成再转换
        arr = numpy.random.random(shape).astype(dtype)
        arr = arr * scale
        return xp.ndarray.from_numpy(arr)


def shaped_reverse_arange(shape, dtype=numpy.float64, order='C', xp=None):
    """生成指定形状的反向序列数组
    
    生成从大到小的序列，然后reshape成指定形状。
    用于测试降序数据的处理。
    """
    if xp is None:
        xp = numpy
    
    # 处理shape参数
    if isinstance(shape, int):
        shape = (shape,)
    
    # 计算总元素数
    size = 1
    for dim in shape:
        size *= dim
    
    # 生成反向序列
    if xp is numpy:
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        return arr.reshape(shape, order=order)
    else:
        # 对于asnumpy，先用numpy生成再转换
        arr = numpy.arange(size - 1, -1, -1, dtype=dtype)
        arr = arr.reshape(shape, order=order)
        return xp.ndarray.from_numpy(arr)


def assert_array_list_equal(x_list, y_list, err_msg='', verbose=True):
    """比较两个数组列表是否相等
    
    用于测试返回多个数组的函数。
    """
    from . import _array
    
    if len(x_list) != len(y_list):
        raise AssertionError(f"List lengths differ: {len(x_list)} vs {len(y_list)}")
    
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        try:
            _array.assert_array_equal(x, y, err_msg, verbose)
        except AssertionError as e:
            raise AssertionError(f"Arrays at index {i} differ: {e}") from e


def suppress_warnings(func):
    """装饰器：抑制函数执行时的警告
    
    用于测试中临时忽略已知的警告。
    
    Examples:
        @suppress_warnings
        def test_something():
            # 这里的警告会被忽略
            pass
    """
    import warnings
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return func(*args, **kwargs)
    return wrapper


def with_seed(seed):
    """装饰器：使用固定随机种子运行测试
    
    确保测试的可重现性。
    
    Args:
        seed: 随机种子
        
    Examples:
        @with_seed(42)
        def test_random_function():
            arr = numpy.random.random((3, 3))
            # 每次运行都会得到相同的随机数
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 保存当前随机状态
            old_state = numpy.random.get_state()
            try:
                # 设置新的随机种子
                numpy.random.seed(seed)
                return func(*args, **kwargs)
            finally:
                # 恢复原来的随机状态
                numpy.random.set_state(old_state)
        return wrapper
    return decorator


def generate_test_data(func):
    """装饰器：为测试函数自动生成测试数据
    
    根据函数签名自动生成常见的测试用例。
    这是一个简化的实现，可以根据需要扩展。
    
    Examples:
        @generate_test_data
        def test_add(a, b):
            return a + b
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 这里可以实现自动生成测试数据的逻辑
        # 简化版本：直接调用原函数
        return func(*args, **kwargs)
    return wrapper


# 一些常用的测试常量
TEST_SHAPES = [
    (),           # 标量
    (0,),         # 空数组
    (1,),         # 单元素
    (5,),         # 一维
    (2, 3),       # 二维
    (2, 3, 4),    # 三维
    (1, 2, 3, 4), # 四维
]

TEST_DTYPES = [
    numpy.float32,
    numpy.float64,
    numpy.int32,
    numpy.int64,
]

TEST_ORDERS = ['C', 'F']