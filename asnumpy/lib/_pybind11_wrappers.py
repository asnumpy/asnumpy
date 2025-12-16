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

