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

"""参数化测试工具

提供更复杂的参数化测试功能：
- product() - 生成参数的笛卡尔积
- _make_class_name() - 为参数化测试生成类名
"""

__all__ = [
    'product',
    'product_dict',
    'parameterize_test_class',
]

import itertools


def product(params_dict):
    """生成参数字典的笛卡尔积
    
    将多个参数的可能值组合成笛卡尔积，用于参数化测试。
    
    Args:
        params_dict: 参数字典，键是参数名，值是参数可能的取值列表
        
    Returns:
        生成器，产生参数组合的字典
        
    Examples:
        >>> list(product({'a': [1, 2], 'b': [3, 4]}))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    if not params_dict:
        yield {}
        return
    
    keys = list(params_dict.keys())
    values = list(params_dict.values())
    
    for combination in itertools.product(*values):
        yield dict(zip(keys, combination))


def product_dict(*dicts):
    """将多个字典的笛卡尔积合并
    
    Args:
        *dicts: 多个参数字典
        
    Returns:
        生成器，产生合并后的参数字典
        
    Examples:
        >>> list(product_dict({'a': [1, 2]}, {'b': [3, 4]}))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    # 合并所有字典
    merged = {}
    for d in dicts:
        merged.update(d)
    
    return product(merged)


def _make_class_name(base_name, params):
    """为参数化测试生成类名
    
    根据基础类名和参数值，生成一个描述性的测试类名。
    
    Args:
        base_name: 基础类名
        params: 参数字典
        
    Returns:
        生成的类名字符串
        
    Examples:
        >>> _make_class_name('TestZeros', {'dtype': 'float32', 'order': 'C'})
        'TestZeros_dtype_float32_order_C'
    """
    if not params:
        return base_name
    
    parts = [base_name]
    for key, value in params.items():
        # 将参数值转换为字符串，并清理特殊字符
        value_str = str(value)
        # 移除类型前缀（如 <class 'numpy.float32'> -> float32）
        if 'numpy.' in value_str:
            value_str = value_str.split('.')[-1].rstrip("'>")
        # 替换特殊字符
        value_str = value_str.replace('<', '').replace('>', '').replace("'", '')
        value_str = value_str.replace(' ', '_').replace('.', '_').replace('-', '_')
        
        parts.append(f'{key}_{value_str}')
    
    return '_'.join(parts)


def parameterize_test_class(base_class, params_dict):
    """为测试类生成参数化的子类
    
    这个函数根据参数字典，为基础测试类生成多个参数化的子类。
    
    Args:
        base_class: 基础测试类
        params_dict: 参数字典
        
    Returns:
        生成的测试类列表
        
    Examples:
        class BaseTest:
            def test_func(self):
                pass
        
        classes = parameterize_test_class(
            BaseTest,
            {'dtype': [numpy.float32, numpy.float64]}
        )
    """
    classes = []
    
    for params in product(params_dict):
        # 生成类名
        class_name = _make_class_name(base_class.__name__, params)
        
        # 创建新类
        new_class = type(
            class_name,
            (base_class,),
            {'_params': params}
        )
        
        classes.append(new_class)
    
    return classes

