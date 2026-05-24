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

"""Parameterized test utilities.

Provides advanced parameterized test functionality:
- product() - generate Cartesian product of parameters
- _make_class_name() - generate class names for parameterized tests
"""

__all__ = [
    "product",
    "product_dict",
    "parameterize_test_class",
]

import itertools


def product(params_dict):
    """Generate the Cartesian product of a parameter dictionary.

    Combines all possible values of multiple parameters into a Cartesian
    product for use in parameterized tests.

    Args:
        params_dict: Dict mapping parameter names to lists of possible values.

    Returns:
        Generator yielding dicts of parameter combinations.

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
        yield dict(zip(keys, combination, strict=False))


def product_dict(*dicts):
    """Merge multiple dicts and yield their Cartesian product.

    Args:
        *dicts: Multiple parameter dicts.

    Returns:
        Generator yielding merged parameter dicts.

    Examples:
        >>> list(product_dict({'a': [1, 2]}, {'b': [3, 4]}))
        [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    """
    merged = {}
    for d in dicts:
        merged.update(d)

    return product(merged)


def _make_class_name(base_name, params):
    """Generate a class name for a parameterized test.

    Builds a descriptive test class name from a base name and parameter values.

    Args:
        base_name: Base class name.
        params: Parameter dict.

    Returns:
        Generated class name string.

    Examples:
        >>> _make_class_name('TestZeros', {'dtype': 'float32', 'order': 'C'})
        'TestZeros_dtype_float32_order_C'
    """
    if not params:
        return base_name

    parts = [base_name]
    for key, value in params.items():
        value_str = str(value)
        # Strip type prefix (e.g. <class 'numpy.float32'> -> float32)
        if "numpy." in value_str:
            value_str = value_str.split(".")[-1].rstrip("'>")
        # Replace special characters
        value_str = value_str.replace("<", "").replace(">", "").replace("'", "")
        value_str = value_str.replace(" ", "_").replace(".", "_").replace("-", "_")

        parts.append(f"{key}_{value_str}")

    return "_".join(parts)


def parameterize_test_class(base_class, params_dict):
    """Generate parameterized subclasses for a test class.

    Creates multiple parameterized subclasses of a base test class
    according to the given parameter dict.

    Args:
        base_class: Base test class.
        params_dict: Parameter dict.

    Returns:
        List of generated test classes.

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
        # Generate class name
        class_name = _make_class_name(base_class.__name__, params)

        # Create new class
        new_class = type(class_name, (base_class,), {"_params": params})

        classes.append(new_class)

    return classes
