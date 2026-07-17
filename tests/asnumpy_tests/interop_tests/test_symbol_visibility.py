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

"""Lock the symbol-visibility policy of the built _core extension.

asnumpy's public ABI is exactly one symbol: PyInit__core. Nothing else is linkable (every csrc/
target is an OBJECT lib, no header is installed, the wheel ships only the .so), so every C++ symbol
is an implementation detail and must stay hidden. `CMAKE_CXX_VISIBILITY_PRESET hidden` +
`CMAKE_VISIBILITY_INLINES_HIDDEN ON` in the root CMakeLists.txt enforce that.

Without them _core.so exported 1027 dynamic symbols: 808 of them fmt::/spdlog:: vague-linkage
template copies, 26 of those *weak* and name-identical to the libfmt.so.8 / libspdlog.so.1 that
_core.so dynamically links. Weak symbols merge by name at load time, first one wins -- so another
extension in the same interpreter with a different fmt/spdlog could bind our calls to its copy.

These tests assert the *intent* (nothing leaks) rather than an exact symbol count. The residual
symbols are libstdc++ types deliberately marked _GLIBCXX_VISIBILITY(default), and their number
drifts with the GCC and pybind11 versions -- pinning a total would produce false failures.

Linux/ELF only, which matches the project's supported-platform set (pyproject.toml declares
"Operating System :: POSIX :: Linux" and nothing else).
"""

import shutil
import subprocess
import sys

import pytest

import asnumpy


def _exported_symbols() -> list[str]:
    """Demangled names of the dynamic symbols _core.so defines."""
    so = asnumpy._core.__file__
    out = subprocess.run(
        ["nm", "-D", "--defined-only", "-C", so],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.splitlines()


pytestmark = [
    pytest.mark.skipif(shutil.which("nm") is None, reason="binutils nm not available"),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ELF/nm specific"),
]


def test_module_init_symbol_is_exported():
    """PyInit__core is the one symbol that must be exported, or Python cannot load the module."""
    assert any("PyInit__core" in line for line in _exported_symbols())


@pytest.mark.parametrize("leaked", ["fmt::", "spdlog::"])
def test_bundled_cxx_dependencies_do_not_leak(leaked):
    """fmt/spdlog template copies must not be exported.

    They are weak and collide by name with the libfmt.so.8 / libspdlog.so.1 that _core.so links,
    so exporting them lets another extension's copy win at load time.
    """
    hits = [line for line in _exported_symbols() if leaked in line]
    assert not hits, f"{len(hits)} {leaked} symbols exported, e.g. {hits[:3]}"


def test_asnumpy_types_are_not_exported():
    """asnumpy's own types are implementation details, not an ABI.

    NPUArray matters most: it sits at global namespace scope, so an exported typeinfo could be
    interposed against another DSO defining its own `NPUArray`.
    """
    symbols = _exported_symbols()
    assert not [s for s in symbols if "NPUArray" in s]
    assert not [s for s in symbols if "asnumpy::" in s]
