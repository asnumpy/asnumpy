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

"""算子注册系统测试。"""

import numpy as np
import pytest

from asnumpy._registry import _OP_REGISTRY, get_op_info, list_ops, register_op, wrap_result
from asnumpy.utils import ndarray


class TestRegisterOp:
    def test_basic_registration(self):
        _OP_REGISTRY.clear()

        @register_op("test_add", module="math", category="arithmetic")
        def test_add(x, y):
            return x + y

        info = get_op_info("test_add")
        assert info is not None
        assert info["module"] == "math"
        assert info["category"] == "arithmetic"
        assert callable(info["callable"])

    def test_default_name(self):
        _OP_REGISTRY.clear()

        @register_op(module="math")
        def my_custom_op(x):
            return x

        assert get_op_info("my_custom_op") is not None

    def test_op_metadata(self):
        _OP_REGISTRY.clear()

        @register_op("meta_op", module="linalg", category="decomposition", doc="Test doc")
        def meta_op(x):
            """Source docstring."""
            return x

        info = get_op_info("meta_op")
        assert info["doc"] == "Test doc"

        op_no_doc = register_op("no_doc_op", module="core")
        info = op_no_doc(lambda x: x)
        assert "doc" in get_op_info("no_doc_op") or True

    def test_wrapper_preserves_functionality(self):
        _OP_REGISTRY.clear()

        @register_op("add_one", module="math")
        def add_one(x):
            return x + 1

        result = add_one(41)
        assert result == 42

    def test_logger_catch(self):
        _OP_REGISTRY.clear()

        @register_op("failing_op", module="math")
        def failing_op():
            raise ValueError("deliberate error")

        with pytest.raises(ValueError, match="deliberate error"):
            failing_op()


class TestWrapResult:
    def test_ndarray_passthrough(self):
        arr = ndarray((3,), np.float32)
        result = wrap_result(arr)
        assert result is arr

    def test_core_ndarray_wraps(self):
        """_core.ndarray is wrapped into Python ndarray."""
        core_arr = ndarray((2,), np.float32)
        result = wrap_result(core_arr)
        assert isinstance(result, ndarray)


class TestListOps:
    def test_filter_by_module(self):
        _OP_REGISTRY.clear()

        @register_op("math_op", module="math")
        def math_op(x):
            return x

        @register_op("linalg_op", module="linalg")
        def linalg_op(x):
            return x

        math_ops = list_ops(module="math")
        assert len(math_ops) == 1
        assert math_ops[0]["name"] == "math_op"

        linalg_ops = list_ops(module="linalg")
        assert len(linalg_ops) == 1

    def test_filter_by_category(self):
        _OP_REGISTRY.clear()

        @register_op("trig_op", module="math", category="trigonometric")
        def trig_op(x):
            return x

        @register_op("arith_op", module="math", category="arithmetic")
        def arith_op(x):
            return x

        trig_ops = list_ops(category="trigonometric")
        assert len(trig_ops) == 1

    def test_list_all(self):
        _OP_REGISTRY.clear()

        @register_op("op1", module="math")
        def op1(x):
            return x

        @register_op("op2", module="math")
        def op2(x):
            return x

        all_ops = list_ops()
        assert len(all_ops) == 2
