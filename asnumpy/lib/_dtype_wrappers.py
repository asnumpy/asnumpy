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
可调用的 dtype 包装类，使其同时支持：
1. 调用创建标量: ap.bfloat16(3.14)
2. 直接作为 dtype: dtype=ap.bfloat16
3. 用于结构化数组: np.dtype([('x', ap.bfloat16)])

关键：让 _CallableDtype 在类型检查时被视为 np.dtype
"""
import numpy as np
import types


class _CallableDtype:
    """
    包装 dtype 对象和类型对象，使其同时具有：
    - dtype 对象的功能（可以作为 dtype 使用）
    - 类型对象的功能（可以调用创建标量）
    
    关键特性：
    - 在 isinstance() 检查时被视为 np.dtype
    - 在传递给 pybind11 时自动转换为 np.dtype
    """
    
    def __init__(self, dtype_obj, type_obj):
        """
        初始化可调用的 dtype 包装对象
        
        Args:
            dtype_obj: numpy.dtype 对象
            type_obj: 类型对象（可以调用创建标量）
        """
        object.__setattr__(self, '_dtype', dtype_obj)
        object.__setattr__(self, '_type', type_obj)
        # 关键：在对象创建后直接修改 __class__ 属性
        # 这会让 isinstance() 检查通过，因为 isinstance() 会直接访问 __class__
        # 注意：必须在所有属性设置完成后才能修改 __class__
        # 使用 object.__setattr__ 来绕过常规的属性设置机制
        try:
            # 直接修改对象的 __class__ 属性
            # 这会让 isinstance(obj, np.dtype) 返回 True
            # 注意：在某些 Python 版本中，可能需要使用 self.__dict__['__class__'] = np.dtype
            object.__setattr__(self, '__class__', np.dtype)
        except (TypeError, AttributeError) as e:
            # 如果无法直接设置 __class__（某些 Python 版本或 C 扩展类型可能不允许），
            # 尝试使用 __dict__ 直接修改
            try:
                self.__dict__['__class__'] = np.dtype
            except (TypeError, AttributeError):
                # 如果还是不行，我们依赖 __getattribute__ 来拦截 __class__ 访问
                pass
    
    def __call__(self, value):
        """调用类型对象创建标量"""
        return self._type(value)
    
    def __array_dtype__(self):
        """
        NumPy 1.20+ 支持的方法，用于在数组创建时自动转换
        返回底层的 dtype 对象
        """
        return self._dtype
    
    def __getattr__(self, name):
        """代理所有属性访问到底层的 dtype 对象"""
        return getattr(self._dtype, name)
    
    def __repr__(self):
        """返回 dtype 对象的字符串表示"""
        return repr(self._dtype)
    
    def __str__(self):
        """返回 dtype 对象的字符串表示"""
        return str(self._dtype)
    
    def __eq__(self, other):
        """比较是否相等"""
        if isinstance(other, _CallableDtype):
            return self._dtype == other._dtype
        return self._dtype == other
    
    def __hash__(self):
        """支持作为字典键"""
        return hash(self._dtype)
    
    # 让 np.dtype() 能够处理这个对象
    # 注意：np.dtype() 构造函数会检查 isinstance(obj, np.dtype)
    # 如果返回 True，会直接使用对象；否则会尝试转换
    # 由于我们已经让 isinstance() 返回 True，np.dtype() 应该能直接处理
    def __array_function__(self, func, types, args, kwargs):
        """支持 NumPy 的 __array_function__ 协议"""
        if func is np.dtype:
            return self._dtype
        return NotImplemented
    
    # 关键：让 isinstance() 检查通过
    # 重写 __getattribute__ 来拦截对 __class__ 的访问
    def __getattribute__(self, name):
        """重写属性访问，使 isinstance() 检查通过"""
        # 拦截 __class__ 访问，返回 np.dtype
        # 这是为了让 isinstance(obj, np.dtype) 返回 True
        if name == '__class__':
            return np.dtype
        # 对于其他属性，先尝试从对象字典获取
        try:
            # 使用 object.__getattribute__ 来避免递归调用
            return object.__getattribute__(self, name)
        except AttributeError:
            # 如果对象本身没有该属性，尝试从 dtype 对象获取
            # 注意：这里也需要使用 object.__getattribute__ 来避免递归
            try:
                dtype = object.__getattribute__(self, '_dtype')
                return getattr(dtype, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    # 兼容性：让对象在某些情况下自动转换为 dtype
    @property
    def dtype(self):
        """返回底层的 dtype 对象"""
        return self._dtype

