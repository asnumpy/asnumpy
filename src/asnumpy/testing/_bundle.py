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

"""Test class generation utilities.

Responsible for dynamically generating test classes:
- make_decorator() - create decorators
- _generate_case() - generate concrete test classes
"""

__all__ = [
    "make_decorator",
    "generate_test_classes",
    "TestBundle",
]

import functools


def make_decorator(decorator_func):
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    """General-purpose utility for creating decorators.

    Converts a plain function into a decorator.

    Args:
        decorator_func: The decorator function.
=======
    """创建装饰器的通用工具

    这个函数可以将一个普通函数转换为装饰器。

    Args:
        decorator_func: 装饰器函数
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

    Returns:
        Decorator.
    """

    @functools.wraps(decorator_func)
    def wrapper(*args, **kwargs):
        # Used directly as @decorator
        if len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            return decorator_func(func)
        # Used as @decorator(...)
        else:

            def actual_decorator(func):
                return decorator_func(func, *args, **kwargs)

            return actual_decorator

    return wrapper


def _generate_case(test_class, params):
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    """Generate a concrete test case class.

    Creates a concrete test case class for a given combination of parameters.

    Args:
        test_class: Base test class.
        params: Parameter dict.
=======
    """生成具体的测试用例类

    根据参数组合，为测试类生成一个具体的测试用例类。

    Args:
        test_class: 基础测试类
        params: 参数字典
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

    Returns:
        Generated test class.
    """
    # Build class name
    class_name = test_class.__name__
    for key, value in params.items():
        value_str = str(value)
        if hasattr(value, "__name__"):
            value_str = value.__name__
        class_name += f"_{key}_{value_str}"

<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    # Build attribute dict for the new class
    class_dict = {}

    # Copy all methods from the test class
=======
    # 创建新类的属性字典
    class_dict = {}

    # 复制测试类的所有方法
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py
    for attr_name in dir(test_class):
        if attr_name.startswith("_"):
            continue
        attr = getattr(test_class, attr_name)
        if not callable(attr):
            continue

<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
        # Bind parameters into each method
=======
        # 为方法添加参数
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py
        def make_method(original_method, test_params):
            @functools.wraps(original_method)
            def method(self, *args, **kwargs):
                kwargs.update(test_params)
                return original_method(self, *args, **kwargs)

            return method

        class_dict[attr_name] = make_method(attr, params)

<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    # Create the new class
=======
    # 创建新类
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py
    new_class = type(class_name, (test_class,), class_dict)
    return new_class


def generate_test_classes(base_class, param_combinations):
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    """Generate multiple parameterized versions of a test class.

    Args:
        base_class: Base test class.
        param_combinations: List of parameter combination dicts.
=======
    """为测试类生成多个参数化版本

    Args:
        base_class: 基础测试类
        param_combinations: 参数组合列表
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

    Returns:
        List of generated test classes.
    """
    test_classes = []

    for params in param_combinations:
        test_class = _generate_case(base_class, params)
        test_classes.append(test_class)

    return test_classes


class TestBundle:
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
    """Wrapper class for a collection of test classes.

    Wraps multiple test classes for convenient batch management.
    """

    def __init__(self, test_classes):
        """Initialize the test bundle.
=======
    """测试包装类

    这个类可以包装多个测试类，方便批量管理。
    """

    def __init__(self, test_classes):
        """初始化测试包装类
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

        Args:
            test_classes: List of test classes.
        """
        self.test_classes = test_classes

    def run(self, runner):
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
        """Run all test classes.

        Args:
            runner: Test runner.
=======
        """运行所有测试类

        Args:
            runner: 测试运行器
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

        Returns:
            List of test results.
        """
        results = []
        for test_class in self.test_classes:
            suite = runner.loadTestsFromTestCase(test_class)
            result = runner.run(suite)
            results.append(result)
        return results

    def get_test_count(self):
<<<<<<< HEAD:src/asnumpy/testing/_bundle.py
        """Return the total number of test methods.
=======
        """获取测试数量
>>>>>>> 6f1d96a (style: fix ruff formatting for project python files):asnumpy/testing/_bundle.py

        Returns:
            Total count of test methods across all classes.
        """
        count = 0
        for test_class in self.test_classes:
            for attr_name in dir(test_class):
                if attr_name.startswith("test_"):
                    count += 1
        return count
