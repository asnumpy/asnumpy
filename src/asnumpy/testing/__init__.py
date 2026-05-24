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
    # Array assertions
    "assert_array_equal",
    "assert_allclose",
    "assert_array_list_equal",
    # Exception and warning assertions
    "assert_raises",
    "assert_raises_regex",
    "assert_warns",
    "assert_no_warnings",
    "assert_equal",
    "assert_string_equal",
    "assert_warns_message",
    # dtype decorators
    "for_dtypes",
    "for_all_dtypes",
    "for_float_dtypes",
    "for_int_dtypes",
    "for_signed_dtypes",
    "for_unsigned_dtypes",
    "for_complex_dtypes",
    # order decorators
    "for_orders",
    "for_cf_orders",
    # numpy-asnumpy comparison decorators
    "numpy_asnumpy_array_equal",
    "numpy_asnumpy_allclose",
    # pytest integration
    "pytest_is_available",
    "parameterize",
    "fixture",
    "skip",
    "skipif",
    "xfail",
    # parameterization utilities
    "product",
    "product_dict",
    "parameterize_test_class",
    # test class generation
    "make_decorator",
    "generate_test_classes",
    "TestBundle",
    # helper functions
    "shaped_arange",
    "shaped_random",
    "shaped_reverse_arange",
    "suppress_warnings",
    "with_seed",
    "generate_test_data",
    # test constants
    "TEST_SHAPES",
    "TEST_DTYPES",
    "TEST_ORDERS",
]

# Array assertion functions
from asnumpy.testing._array import assert_allclose, assert_array_equal

# Exception and warning assertions
from asnumpy.testing._assertions import (
    assert_equal,
    assert_no_warnings,
    assert_raises,
    assert_raises_regex,
    assert_string_equal,
    assert_warns,
    assert_warns_message,
)

# Test class generation utilities
from asnumpy.testing._bundle import (
    TestBundle,
    generate_test_classes,
    make_decorator,
)

# Test helper functions
from asnumpy.testing._helper import (
    TEST_DTYPES,
    TEST_ORDERS,
    TEST_SHAPES,
    assert_array_list_equal,
    generate_test_data,
    shaped_arange,
    shaped_random,
    shaped_reverse_arange,
    suppress_warnings,
    with_seed,
)

# dtype and order parameterization decorators
from asnumpy.testing._loops import (
    for_all_dtypes,
    for_cf_orders,
    for_complex_dtypes,
    for_dtypes,
    for_float_dtypes,
    for_int_dtypes,
    for_orders,
    for_signed_dtypes,
    for_unsigned_dtypes,
    numpy_asnumpy_allclose,
    numpy_asnumpy_array_equal,
)

# Parameterization utilities
from asnumpy.testing._parameterized import (
    parameterize_test_class,
    product,
    product_dict,
)
from asnumpy.testing._pytest_impl import (
    fixture,
    parameterize,
    skip,
    skipif,
    xfail,
)

# pytest integration
from asnumpy.testing._pytest_impl import (
    is_available as pytest_is_available,
)
