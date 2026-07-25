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

These tests assert the *intent* (no leaked *code* symbols) rather than an exact symbol count. The
residual symbols are libstdc++ types deliberately marked _GLIBCXX_VISIBILITY(default), and their
number drifts with the GCC and pybind11 versions -- pinning a total would produce false failures.

RTTI (typeinfo / typeinfo name / vtable) for some third-party template instantiations can also stay
default-visible even with -fvisibility=hidden: e.g. Ubuntu noble's libspdlog 1.12 shared library
exports ``spdlog::sinks::base_sink<std::mutex>`` RTTI because ``std::mutex`` carries
``_GLIBCXX_VISIBILITY(default)``. Those three symbols are required for cross-DSO typeinfo dedup
against the libspdlog.so we link; they are not the weak function copies this test is meant to
catch. Filter them out of the fmt/spdlog leak check. ``test_asnumpy_types_are_not_exported`` stays
strict -- an exported NPUArray typeinfo is exactly the hazard that test guards against.

Linux/ELF only, which matches the project's supported-platform set (pyproject.toml declares
"Operating System :: POSIX :: Linux" and nothing else).
"""

import importlib
import shutil
import subprocess
import sys

import pytest

_nm = shutil.which("nm")

# Demangled nm -C lines: " typeinfo for Foo", " typeinfo name for Foo", " vtable for Foo".
_RTTI_MARKERS = (" typeinfo for ", " typeinfo name for ", " vtable for ")


def _exported_symbols() -> list[str]:
    """Demangled names of the dynamic symbols _core.so defines."""
    assert _nm is not None
    so = importlib.import_module("asnumpy._core").__file__
    out = subprocess.run(
        [_nm, "-D", "--defined-only", "-C", so],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.splitlines()


def _is_rtti(line: str) -> bool:
    """RTTI must stay default-visibility to dedup against the DSO we link."""
    return any(marker in line for marker in _RTTI_MARKERS)


pytestmark = [
    pytest.mark.skipif(_nm is None, reason="binutils nm not available"),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="ELF/nm specific"),
]


def test_module_init_symbol_is_exported():
    """PyInit__core is the one symbol that must be exported, or Python cannot load the module."""
    assert any("PyInit__core" in line for line in _exported_symbols())


@pytest.mark.parametrize("leaked", ["fmt::", "spdlog::"])
def test_bundled_cxx_dependencies_do_not_leak(leaked):
    """fmt/spdlog template *code* copies must not be exported.

    They are weak and collide by name with the libfmt.so.8 / libspdlog.so.1 that _core.so links,
    so exporting them lets another extension's copy win at load time. RTTI symbols are excluded;
    see module docstring.
    """
    hits = [line for line in _exported_symbols() if leaked in line and not _is_rtti(line)]
    assert not hits, f"{len(hits)} {leaked} symbols exported, e.g. {hits[:3]}"


def test_asnumpy_types_are_not_exported():
    """asnumpy's own types are implementation details, not an ABI.

    NPUArray matters most: it sits at global namespace scope, so an exported typeinfo could be
    interposed against another DSO defining its own `NPUArray`.
    """
    symbols = _exported_symbols()
    assert not [s for s in symbols if "NPUArray" in s]
    assert not [s for s in symbols if "asnumpy::" in s]
