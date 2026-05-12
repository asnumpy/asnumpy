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

"""pytest集成实现

这个文件处理pytest和Asnumpy测试框架的集成：
- parameterize() - 参数化测试的实现
- _TestingParameterizeMixin - 参数化测试的混合类
- is_available() - 检查pytest是否可用
"""

__all__ = [
    'is_available',
    'parameterize',
    'fixture',
    'skip',
    'skipif',
    'xfail',
    '_TestingParameterizeMixin',
]

import functools


def is_available():
    """检查pytest是否可用
    
    Returns:
        bool: pytest是否已安装并可用
    """
    try:
        import pytest
        return True
    except ImportError:
        return False


class _TestingParameterizeMixin:
    """参数化测试的混合类
    
    这个类可以被测试类继承，以支持pytest风格的参数化测试。
    """
    
    @classmethod
    def setup_class(cls):
        """在测试类开始前执行的设置"""
        pass
    
    @classmethod
    def teardown_class(cls):
        """在测试类结束后执行的清理"""
        pass


def parameterize(*args, **kwargs):
    """参数化测试装饰器
    
    这个装饰器提供类似pytest.mark.parametrize的功能，
    但与Asnumpy的测试框架集成。
    
    Args:
        *args: 参数名和参数值
        **kwargs: 其他选项
        
    Returns:
        装饰器函数
        
    Examples:
        @parameterize('dtype', [numpy.float32, numpy.float64])
        def test_func(self, dtype):
            ...
    """
    if not is_available():
        # 如果pytest不可用，使用简单的循环实现
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *func_args, **func_kwargs):
                if len(args) >= 2:
                    param_name = args[0]
                    param_values = args[1]
                    for value in param_values:
                        func_kwargs[param_name] = value
                        func(self, *func_args, **func_kwargs)
                else:
                    func(self, *func_args, **func_kwargs)
            return wrapper
        return decorator
    
    # 如果pytest可用，使用pytest.mark.parametrize
    import pytest
    return pytest.mark.parametrize(*args, **kwargs)


def fixture(*args, **kwargs):
    """fixture装饰器
    
    这个装饰器提供类似pytest.fixture的功能。
    
    Args:
        *args: 位置参数
        **kwargs: 关键字参数
        
    Returns:
        装饰器函数或fixture对象
    """
    if not is_available():
        # 如果pytest不可用，返回一个简单的装饰器
        if len(args) == 1 and callable(args[0]):
            # 直接作为@fixture使用
            return args[0]
        else:
            # 作为@fixture(...)使用
            def decorator(func):
                return func
            return decorator
    
    # 如果pytest可用，使用pytest.fixture
    import pytest
    return pytest.fixture(*args, **kwargs)


def skip(reason):
    """跳过测试装饰器
    
    Args:
        reason: 跳过测试的原因
        
    Returns:
        装饰器函数
    """
    if not is_available():
        import unittest
        return unittest.skip(reason)
    
    import pytest
    return pytest.mark.skip(reason=reason)


def skipif(condition, reason):
    """条件跳过测试装饰器
    
    Args:
        condition: 跳过测试的条件
        reason: 跳过测试的原因
        
    Returns:
        装饰器函数
    """
    if not is_available():
        import unittest
        return unittest.skipIf(condition, reason)
    
    import pytest
    return pytest.mark.skipif(condition, reason=reason)


def xfail(reason='', strict=False):
    """预期失败测试装饰器
    
    Args:
        reason: 预期失败的原因
        strict: 是否严格模式
        
    Returns:
        装饰器函数
    """
    if not is_available():
        import unittest
        return unittest.expectedFailure
    
    import pytest
    return pytest.mark.xfail(reason=reason, strict=strict)