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
"""
Pybind11 函数和类的包装模块

此模块提供包装函数和类，使 _CallableDtype 能够在传递给 pybind11 绑定的函数和类之前
自动转换为 numpy.dtype。这样可以让用户使用 ap.bfloat16 等可调用的 dtype 对象，
同时保持与 pybind11 的兼容性。
"""
import numpy as np
from ._dtype_wrappers import _CallableDtype


def convert_dtype_for_pybind11(dtype):
    """
    将 _CallableDtype 对象转换为 numpy.dtype，以便 pybind11 能够识别。
    
    Args:
        dtype: 可能是 _CallableDtype 或 numpy.dtype 对象
        
    Returns:
        numpy.dtype 对象
    """
    if isinstance(dtype, _CallableDtype):
        return dtype._dtype
    return dtype


def wrap_array_creation_functions(ones_func, zeros_func, empty_func, full_func, 
                                   eye_func, identity_func, ones_like_func, 
                                   zeros_like_func, full_like_func, empty_like_func):
    """
    包装数组创建函数，使其支持 _CallableDtype。
    
    Args:
        ones_func, zeros_func, ...: 原始的 pybind11 函数
        
    Returns:
        dict: 包含包装后的函数的字典
    """
    def wrap_ones(original_func):
        def wrapper(shape, dtype):
            return original_func(shape, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_zeros(original_func):
        def wrapper(shape, dtype):
            return original_func(shape, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_empty(original_func):
        def wrapper(shape, dtype):
            return original_func(shape, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_full(original_func):
        def wrapper(shape, value, dtype):
            return original_func(shape, value, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_eye(original_func):
        def wrapper(n, dtype):
            return original_func(n, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_identity(original_func):
        def wrapper(n, dtype):
            return original_func(n, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_ones_like(original_func):
        def wrapper(other, dtype):
            return original_func(other, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_zeros_like(original_func):
        def wrapper(other, dtype):
            return original_func(other, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_full_like(original_func):
        def wrapper(other, value, dtype):
            return original_func(other, value, convert_dtype_for_pybind11(dtype))
        return wrapper
    
    def wrap_empty_like(original_func):
        def wrapper(prototype, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(prototype, dtype)
        return wrapper
    
    return {
        'ones': wrap_ones(ones_func),
        'zeros': wrap_zeros(zeros_func),
        'empty': wrap_empty(empty_func),
        'full': wrap_full(full_func),
        'eye': wrap_eye(eye_func),
        'identity': wrap_identity(identity_func),
        'ones_like': wrap_ones_like(ones_like_func),
        'zeros_like': wrap_zeros_like(zeros_like_func),
        'full_like': wrap_full_like(full_like_func),
        'empty_like': wrap_empty_like(empty_like_func),
    }


def wrap_ndarray_class(ndarray_class):
    """
    包装 ndarray 类，使其构造函数支持 _CallableDtype。
    
    Args:
        ndarray_class: 原始的 pybind11 绑定的 ndarray 类
        
    Returns:
        包装后的类，支持 _CallableDtype
    """
    class NPUArrayWrapper:
        """
        包装 NPUArray 类以支持 _CallableDtype。
        
        这个类包装了原始的 ndarray 类，在调用构造函数前将 _CallableDtype 转换为 numpy.dtype。
        同时保留所有原始类的功能（静态方法、实例方法等）。
        """
        
        def __new__(cls, shape=None, dtype=None, *args, **kwargs):
            """
            创建 NPUArray 实例，支持 _CallableDtype。
            
            Args:
                shape: 数组形状
                dtype: 数据类型（可以是 _CallableDtype 或 numpy.dtype）
                *args, **kwargs: 其他参数（用于复制构造函数等）
            
            Returns:
                NPUArray 实例
            """
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            
            if shape is not None:
                return ndarray_class(shape, dtype, *args, **kwargs)
            else:
                # 复制构造函数或其他情况
                return ndarray_class(*args, **kwargs)
        
        @staticmethod
        def from_numpy(host_data):
            """从 NumPy 数组创建 NPUArray（静态方法）"""
            return ndarray_class.from_numpy(host_data)
    
    # 使用元类来代理类级别的属性访问
    class NPUArrayMeta(type):
        """元类，用于代理类级别的属性访问"""
        def __getattr__(cls, name):
            """代理所有类属性访问到原始类"""
            return getattr(ndarray_class, name)
    
    # 创建最终的包装类
    class NPUArrayWrapperFinal(NPUArrayWrapper, metaclass=NPUArrayMeta):
        """最终的包装类，支持类级别的属性访问"""
        pass
    
    return NPUArrayWrapperFinal


def wrap_functions_with_optional_dtype(add_func, subtract_func, multiply_func, divide_func,
                                       power_func, maximum_func, minimum_func,
                                       sum_func, prod_func, cumsum_func, cumprod_func):
    """
    包装带有可选 dtype 参数的函数，使其支持 _CallableDtype。
    采用与 wrap_array_creation_functions 相同的方式。
    
    Args:
        add_func, subtract_func, ...: 原始的 pybind11 函数
        
    Returns:
        dict: 包含包装后的函数的字典
    """
    def wrap_add(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_subtract(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_multiply(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_divide(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_power(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_maximum(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_minimum(original_func):
        def wrapper(x1, x2, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(x1, x2, dtype=dtype)
        return wrapper
    
    def wrap_sum(original_func):
        def wrapper(a, axis=None, keepdims=False, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            # sum 有多个重载：一个带 axis/keepdims/dtype，一个只有 a
            if axis is not None:
                return original_func(a, axis, keepdims, dtype)
            else:
                # 如果没有 axis，调用简单版本（不带 dtype）
                return original_func(a)
        return wrapper
    
    def wrap_prod(original_func):
        def wrapper(a, axis=None, keepdims=False, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            # prod 有多个重载：一个带 axis/keepdims/dtype，一个只有 a
            if axis is not None:
                return original_func(a, axis, keepdims, dtype)
            else:
                # 如果没有 axis，调用简单版本（不带 dtype）
                return original_func(a)
        return wrapper
    
    def wrap_cumsum(original_func):
        def wrapper(a, axis, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(a, axis, dtype=dtype)
        return wrapper
    
    def wrap_cumprod(original_func):
        def wrapper(a, axis, dtype=None):
            if dtype is not None:
                dtype = convert_dtype_for_pybind11(dtype)
            return original_func(a, axis, dtype=dtype)
        return wrapper
    
    return {
        'add': wrap_add(add_func),
        'subtract': wrap_subtract(subtract_func),
        'multiply': wrap_multiply(multiply_func),
        'divide': wrap_divide(divide_func),
        'power': wrap_power(power_func),
        'maximum': wrap_maximum(maximum_func),
        'minimum': wrap_minimum(minimum_func),
        'sum': wrap_sum(sum_func),
        'prod': wrap_prod(prod_func),
        'cumsum': wrap_cumsum(cumsum_func),
        'cumprod': wrap_cumprod(cumprod_func),
    }

