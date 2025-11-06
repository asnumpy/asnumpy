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

"""AsNumPy Testing Framework

完整的测试框架，包含：
- 数组断言函数
- 参数化装饰器
- pytest集成
- 测试工具函数
"""

# 数组断言函数
from asnumpy.testing._array import assert_array_equal, assert_allclose

# 装饰器 - dtype和order参数化
from asnumpy.testing._loops import (
    for_dtypes, for_all_dtypes, for_float_dtypes, for_int_dtypes,
    for_signed_dtypes, for_unsigned_dtypes,
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

__all__ = [
    # 数组断言
    'assert_array_equal', 
    'assert_allclose',
    
    # dtype装饰器
    'for_dtypes', 
    'for_all_dtypes', 
    'for_float_dtypes', 
    'for_int_dtypes',
    'for_signed_dtypes', 
    'for_unsigned_dtypes',
    
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
]

