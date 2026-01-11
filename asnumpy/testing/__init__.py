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


__all__ = [
    # 数组断言
    'assert_array_equal', 
    'assert_allclose',
    'assert_array_list_equal',
    
    # 异常和警告断言
    'assert_raises',
    'assert_raises_regex',
    'assert_warns',
    'assert_no_warnings',
    'assert_equal',
    'assert_string_equal',
    'assert_warns_message',
    
    # dtype装饰器
    'for_dtypes', 
    'for_all_dtypes', 
    'for_float_dtypes', 
    'for_int_dtypes',
    'for_signed_dtypes', 
    'for_unsigned_dtypes',
    'for_complex_dtypes',
    
    # order装饰器
    'for_orders', 
    'for_CF_orders',
    
    # numpy-asnumpy比较装饰器
    'numpy_asnumpy_array_equal', 
    'numpy_asnumpy_allclose',
    
    # pytest集成
    'pytest_is_available',
    'parameterize',
    'fixture',
    'skip',
    'skipif',
    'xfail',
    
    # 参数化工具
    'product',
    'product_dict',
    'parameterize_test_class',
    
    # 测试类生成
    'make_decorator',
    'generate_test_classes',
    'TestBundle',
    
    # 辅助函数
    'shaped_arange',
    'shaped_random',
    'shaped_reverse_arange',
    'suppress_warnings',
    'with_seed',
    'generate_test_data',
    
    # 测试常量
    'TEST_SHAPES',
    'TEST_DTYPES',
    'TEST_ORDERS',
]

# 数组断言函数
from asnumpy.testing._array import assert_array_equal, assert_allclose

# 异常和警告断言
from asnumpy.testing._assertions import (
    assert_raises,
    assert_raises_regex,
    assert_warns,
    assert_no_warnings,
    assert_equal,
    assert_string_equal,
    assert_warns_message,
)

# 装饰器 - dtype和order参数化
from asnumpy.testing._loops import (
    for_dtypes, for_all_dtypes, for_float_dtypes, for_int_dtypes,
    for_signed_dtypes, for_unsigned_dtypes, for_complex_dtypes,
    for_orders, for_CF_orders,
    numpy_asnumpy_array_equal, numpy_asnumpy_allclose,
)

# pytest集成
from asnumpy.testing._pytest_impl import (
    is_available as pytest_is_available,
    parameterize,
    fixture,
    skip,
    skipif,
    xfail,
)

# 参数化测试工具
from asnumpy.testing._parameterized import (
    product,
    product_dict,
    parameterize_test_class,
)

# 测试类生成工具
from asnumpy.testing._bundle import (
    make_decorator,
    generate_test_classes,
    TestBundle,
)

# 测试辅助函数
from asnumpy.testing._helper import (
    shaped_arange,
    shaped_random,
    shaped_reverse_arange,
    assert_array_list_equal,
    suppress_warnings,
    with_seed,
    generate_test_data,
    TEST_SHAPES,
    TEST_DTYPES,
    TEST_ORDERS,
)
