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

"""测试循环装饰器

提供用于参数化测试的装饰器，包括dtype、order等维度的循环测试。
"""

import functools
import numpy
from . import _array


# dtype常量定义
_float_dtypes = (numpy.float16, numpy.float32, numpy.float64)
_complex_dtypes = (numpy.complex64, numpy.complex128)
_signed_dtypes = (numpy.int8, numpy.int16, numpy.int32, numpy.int64)
_unsigned_dtypes = (numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64)
_int_dtypes = _signed_dtypes + _unsigned_dtypes


def _make_all_dtypes(no_float16=True, no_bool=False, no_complex=False, no_uint32=True, no_uint64=True):
    """创建所有数据类型列表
    
    默认排除 asnumpy 不支持的类型：float16, uint32, uint64
    
    Args:
        no_float16: 是否排除float16（默认True - 不支持）
        no_bool: 是否排除bool
        no_complex: 是否排除复数类型
        no_uint32: 是否排除uint32（默认True - NPU算子不支持）
        no_uint64: 是否排除uint64（默认True - NPU算子不支持）
        
    Returns:
        包含指定数据类型的元组
    """
    # 浮点类型
    dtypes = list(_float_dtypes)
    if no_float16:
        dtypes.remove(numpy.float16)
    
    # 复数类型
    if not no_complex:
        dtypes.extend(_complex_dtypes)
    
    # 整数类型（包括有符号和无符号）
    int_types = list(_int_dtypes)
    if no_uint32 and numpy.uint32 in int_types:
        int_types.remove(numpy.uint32)
    if no_uint64 and numpy.uint64 in int_types:
        int_types.remove(numpy.uint64)
    dtypes.extend(int_types)
    
    # 布尔类型
    if not no_bool:
        dtypes.append(numpy.bool_)
    
    return tuple(dtypes)


def _wraps_partial(impl, name):
    """函数包装器"""
    def decorator(wrapper):
        return functools.wraps(impl)(wrapper)
    return decorator


# ========== dtype装饰器 ==========

def for_dtypes(dtypes, name='dtype'):
    """为多个数据类型参数化测试
    
    仅支持 pytest 风格（无 self 参数）
    
    Args:
        dtypes: 数据类型列表
        name: 参数名（默认为'dtype'）
        
    Returns:
        装饰器函数
    """
    def decorator(impl):
        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            for dtype in dtypes:
                try:
                    kw[name] = dtype
                    impl(*args, **kw)
                except Exception:
                    print(f'{name} is {dtype}')
                    raise
        return test_func
    return decorator


def for_all_dtypes(name='dtype', no_float16=True, no_bool=False, no_complex=False, no_uint32=True, no_uint64=True):
    """为所有数据类型参数化测试
    
    默认排除 asnumpy 不支持的类型：float16, uint32, uint64
    
    Args:
        name: 参数名
        no_float16: 是否排除float16（默认True - 不支持）
        no_bool: 是否排除bool
        no_complex: 是否排除复数类型
        no_uint32: 是否排除uint32（默认True - NPU算子不支持）
        no_uint64: 是否排除uint64（默认True - NPU算子不支持）
        
    Returns:
        装饰器函数
    """
    return for_dtypes(_make_all_dtypes(no_float16, no_bool, no_complex, no_uint32, no_uint64), name=name)


def for_float_dtypes(name='dtype', no_float16=True):
    """为浮点数数据类型参数化测试
    
    默认排除 float16（asnumpy 不支持）
    
    Args:
        name: 参数名
        no_float16: 是否排除float16（默认True - 不支持）
        
    Returns:
        装饰器函数
    """
    dtypes = list(_float_dtypes)
    if no_float16:
        dtypes.remove(numpy.float16)
    return for_dtypes(tuple(dtypes), name=name)


def for_int_dtypes(name='dtype'):
    """为所有整数类型参数化测试"""
    return for_dtypes(_int_dtypes, name=name)


def for_signed_dtypes(name='dtype'):
    """为有符号整数类型参数化测试"""
    return for_dtypes(_signed_dtypes, name=name)


def for_unsigned_dtypes(name='dtype', no_uint32=True, no_uint64=True):
    """为无符号整数类型参数化测试
    
    默认排除 uint32, uint64（asnumpy NPU算子不支持）
    
    Args:
        name: 参数名
        no_uint32: 是否排除uint32（默认True - NPU算子不支持）
        no_uint64: 是否排除uint64（默认True - NPU算子不支持）
        
    Returns:
        装饰器函数
    """
    dtypes = list(_unsigned_dtypes)
    if no_uint32 and numpy.uint32 in dtypes:
        dtypes.remove(numpy.uint32)
    if no_uint64 and numpy.uint64 in dtypes:
        dtypes.remove(numpy.uint64)
    return for_dtypes(tuple(dtypes), name=name)


# ========== order装饰器 ==========

def for_orders(orders, name='order'):
    """为多个内存顺序参数化测试
    
    仅支持 pytest 风格（无 self 参数）
    
    Args:
        orders: 内存顺序列表
        name: 参数名（默认为'order'）
        
    Returns:
        装饰器函数
    """
    def decorator(impl):
        @_wraps_partial(impl, name)
        def test_func(*args, **kw):
            for order in orders:
                try:
                    kw[name] = order
                    impl(*args, **kw)
                except Exception:
                    print(f'{name} is {order}')
                    raise
        return test_func
    return decorator


def for_CF_orders(name='order'):
    """为C和F内存顺序参数化测试"""
    return for_orders([None, 'C', 'F', 'c', 'f'], name)


# ========== numpy-asnumpy比较装饰器 ==========

def _make_decorator(check_func, name, type_check, accept_error, sp_name=None, scipy_name=None):
    """创建numpy-asnumpy比较装饰器的核心函数
    
    仅支持 pytest 风格（无 self 参数）
    
    Args:
        check_func: 用于比较结果的函数
        name: xp参数名
        type_check: 是否进行类型检查
        accept_error: 是否接受错误
        sp_name: scipy参数名（保留）
        scipy_name: scipy模块名（保留）
        
    Returns:
        装饰器函数
    """
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(*args, **kw):
            # 执行numpy版本
            kw[name] = numpy
            try:
                numpy_result = impl(*args, **kw)
                numpy_error = None
            except Exception as e:
                numpy_result = None
                numpy_error = e
            
            # 执行asnumpy版本
            import asnumpy as ap
            kw[name] = ap
            
            # 为asnumpy自动转换和过滤参数
            kw_asnumpy = kw.copy()
            
            # 转换dtype参数
            if 'dtype' in kw_asnumpy and kw_asnumpy['dtype'] is not None:
                kw_asnumpy['dtype'] = numpy.dtype(kw_asnumpy['dtype'])
            
            # 移除asnumpy不支持的参数（order参数）
            if 'order' in kw_asnumpy:
                kw_asnumpy.pop('order')
            
            try:
                asnumpy_result = impl(*args, **kw_asnumpy)
                asnumpy_error = None
            except Exception as e:
                asnumpy_result = None
                asnumpy_error = e
            
            # 比较结果
            if numpy_error is not None:
                if asnumpy_error is None:
                    raise AssertionError(
                        f'NumPy抛出 {type(numpy_error).__name__}，'
                        f'但AsNumPy没有抛出异常\n'
                        f'NumPy错误: {numpy_error}'
                    )
                elif type(numpy_error) != type(asnumpy_error):
                    if not accept_error:
                        raise AssertionError(
                            f'异常类型不同:\n'
                            f'  NumPy: {type(numpy_error).__name__}\n'
                            f'  AsNumPy: {type(asnumpy_error).__name__}'
                        )
                # 异常类型相同，测试通过
                return
            elif asnumpy_error is not None:
                raise AssertionError(
                    f'AsNumPy抛出 {type(asnumpy_error).__name__}，'
                    f'但NumPy没有抛出异常\n'
                    f'AsNumPy错误: {asnumpy_error}'
                )
            
            # 都没有异常，比较结果
            check_func(numpy_result, asnumpy_result)
        
        return test_func
    return decorator


def numpy_asnumpy_array_equal(err_msg='', verbose=True, name='xp', type_check=True, 
                               accept_error=False, sp_name=None, scipy_name=None, 
                               strides_check=False):
    """装饰器：比较NumPy和AsNumPy的结果是否完全相等
    
    Args:
        err_msg: 错误消息
        verbose: 是否显示详细信息
        name: xp参数名
        type_check: 是否进行类型检查
        accept_error: 是否接受错误
        sp_name: scipy参数名（保留）
        scipy_name: scipy模块名（保留）
        strides_check: 是否检查strides
        
    Returns:
        装饰器函数
    """
    def check_func(x, y):
        _array.assert_array_equal(x, y, err_msg, verbose, strides_check=strides_check)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name, scipy_name)


def numpy_asnumpy_allclose(rtol=1e-7, atol=0, err_msg='', verbose=True, name='xp', 
                            type_check=True, accept_error=False, sp_name=None, 
                            scipy_name=None, strides_check=False):
    """装饰器：比较NumPy和AsNumPy的浮点结果是否在误差范围内
    
    Args:
        rtol: 相对容差
        atol: 绝对容差
        err_msg: 错误消息
        verbose: 是否显示详细信息
        name: xp参数名
        type_check: 是否进行类型检查
        accept_error: 是否接受错误
        sp_name: scipy参数名（保留）
        scipy_name: scipy模块名（保留）
        strides_check: 是否检查strides
        
    Returns:
        装饰器函数
    """
    def check_func(x, y):
        _array.assert_allclose(x, y, rtol, atol, err_msg, verbose, strides_check=strides_check)
    return _make_decorator(check_func, name, type_check, accept_error, sp_name, scipy_name)


__all__ = [
    'for_dtypes', 'for_all_dtypes', 'for_float_dtypes', 'for_int_dtypes',
    'for_signed_dtypes', 'for_unsigned_dtypes',
    'for_orders', 'for_CF_orders',
    'numpy_asnumpy_array_equal', 'numpy_asnumpy_allclose',
]

