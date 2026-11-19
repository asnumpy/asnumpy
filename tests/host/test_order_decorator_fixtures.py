# *****************************************************************************
# Copyright (c) 2026 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

"""Host regression tests for for_orders and pytest fixture collection.

Related issue: for_orders kept the wrapped signature, so pytest tried to
resolve the injected ``order`` (or custom ``name``) parameter as a fixture
and normal tests could not be collected.
"""

import importlib.util
import inspect
import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def _load_loops():
    # Stub the parent packages so the leaf module's relative imports work
    # without initializing the NPU extension. Record the original entries and
    # restore them afterwards so this helper leaves no shells or duplicate
    # module instances behind for the rest of the pytest session.
    injected = {}
    for name in ("asnumpy", "asnumpy.testing", "asnumpy.testing._loops"):
        injected[name] = sys.modules.get(name)
    try:
        for name in ("asnumpy", "asnumpy.testing"):
            if name not in sys.modules:
                package = types.ModuleType(name)
                package.__path__ = [str(_SRC.joinpath(*name.split(".")))]
                package.__spec__ = importlib.util.spec_from_loader(
                    name, loader=None, is_package=True
                )
                sys.modules[name] = package
        spec = importlib.util.spec_from_file_location(
            "asnumpy.testing._loops", _SRC / "asnumpy" / "testing" / "_loops.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["asnumpy.testing._loops"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in injected.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


loops = _load_loops()
for_orders = loops.for_orders


@for_orders(["C", "F"])
def test_for_orders_explicit_param(order):
    """pytest must collect this without treating ``order`` as a fixture."""
    assert order in ("C", "F")


@for_orders(["C", "F"])
def test_for_orders_kw_style(**kw):
    assert kw["order"] in ("C", "F")


def test_signature_hides_injected_param():
    def impl(order, other):
        raise AssertionError("not called")

    decorated = for_orders(["C", "F"])(impl)
    assert "order" not in inspect.signature(decorated).parameters
    assert "other" in inspect.signature(decorated).parameters


def test_signature_hides_custom_name():
    def impl(mode):
        raise AssertionError("not called")

    decorated = for_orders(["C"], name="mode")(impl)
    assert "mode" not in inspect.signature(decorated).parameters


def test_direct_call_still_injects_parameter():
    seen = []

    def impl(mode):
        seen.append(mode)

    decorated = for_orders(["C", "F"], name="mode")(impl)
    decorated()
    assert seen == ["C", "F"]


def test_load_loops_restores_sys_modules():
    names = ("asnumpy", "asnumpy.testing", "asnumpy.testing._loops")
    before = {name: sys.modules.get(name) for name in names}
    _load_loops()
    after = {name: sys.modules.get(name) for name in names}
    assert after == before
