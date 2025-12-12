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

"""数组比较函数

提供用于测试的数组比较断言函数。
"""

__all__ = ['assert_array_equal', 'assert_allclose']

import numpy as np


def assert_array_equal(x, y, err_msg='', verbose=True, strides_check=False):
    """断言两个数组完全相等
    
    Args:
        x: 第一个数组
        y: 第二个数组
        err_msg: 自定义错误消息
        verbose: 是否显示详细错误信息
        strides_check: 是否检查strides
        
    Raises:
        AssertionError: 如果数组不相等
    """
    # 转换为numpy数组
    if not isinstance(x, np.ndarray):
        if hasattr(x, 'to_numpy'):
            x = x.to_numpy()
        else:
            x = np.asarray(x)
    
    if not isinstance(y, np.ndarray):
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        else:
            y = np.asarray(y)
    
    # 检查shape
    if x.shape != y.shape:
        msg = f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查dtype
    if x.dtype != y.dtype:
        msg = f"Dtype mismatch: x.dtype={x.dtype}, y.dtype={y.dtype}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查strides
    if strides_check and x.strides != y.strides:
        msg = f"Strides mismatch: x.strides={x.strides}, y.strides={y.strides}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查值
    if not np.array_equal(x, y):
        if verbose:
            msg = "Arrays are not equal."
            try:
                # 尝试计算差异，仅对数值类型有效
                if x.dtype.kind in 'biu f': # bool, int, uint, float
                    if x.dtype.kind == 'b':
                        # 修复：布尔型使用异或(^)计算差异，避免减法报错
                        diff = x ^ y
                        msg += f"\nNumber of differing elements: {np.sum(diff)}"
                    else:
                        diff = x - y
                        msg += f"\nMax absolute difference: {np.abs(diff).max()}"
            except Exception:
                pass # 如果计算差异失败，忽略，只打印数组内容
            
            if x.size > 0:
                # 找出差异的索引
                msg += f"\nIndices where elements differ: {np.where(x != y)}"
            msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
        else:
            msg = "Arrays are not equal."
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)


def assert_allclose(x, y, rtol=1e-7, atol=0, err_msg='', verbose=True, strides_check=False):
    """断言两个数组在误差范围内相等（用于浮点数比较）"""
    # 转换为numpy数组
    if not isinstance(x, np.ndarray):
        if hasattr(x, 'to_numpy'):
            x = x.to_numpy()
        else:
            x = np.asarray(x)
    
    if not isinstance(y, np.ndarray):
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy()
        else:
            y = np.asarray(y)
    
    # 检查shape
    if x.shape != y.shape:
        msg = f"Shape mismatch: x.shape={x.shape}, y.shape={y.shape}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查dtype
    if x.dtype != y.dtype:
        msg = f"Dtype mismatch: x.dtype={x.dtype}, y.dtype={y.dtype}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查strides
    if strides_check and x.strides != y.strides:
        msg = f"Strides mismatch: x.strides={x.strides}, y.strides={y.strides}"
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)
    
    # 检查值（在误差范围内）
    if not np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
        if verbose:
            msg = "Arrays are not almost equal."
            try:
                if x.dtype.kind in 'uif':
                    diff = x - y
                    msg += f"\nMax absolute difference: {np.abs(diff).max()}"
            except Exception:
                pass

            if x.size > 0:
                mask = ~np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True)
                msg += f"\nIndices where elements differ: {np.where(mask)}"
            msg += f"\nNumPy:\n{x}\nAsNumPy:\n{y}"
        else:
            msg = "Arrays are not almost equal."
        raise AssertionError(f"{err_msg}\n{msg}" if err_msg else msg)