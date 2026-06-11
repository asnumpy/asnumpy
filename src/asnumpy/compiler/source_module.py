# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

"""SourceModule — JIT-compile Ascend C kernel source and expose callable functions.

Analogous to ``pycuda.compiler.SourceModule``.  ~150 lines.

Pipeline:
  source → bisheng → .o → objcopy (.aicore_binary) → rtDevBinaryRegister
         → rtFunctionRegister → ready to launch
"""

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

from . import _rt
from .kernel_function import KernelFunction

# ---------------------------------------------------------------------------
# bisheng compiler helpers
# ---------------------------------------------------------------------------

_ASCEND_HOME = os.environ.get(
    "ASCEND_TOOLKIT_HOME",
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
)

_BISHENG_CANDIDATES = [
    os.path.join(_ASCEND_HOME, "tools", "bisheng_compiler", "bin", "bisheng"),
    os.path.join(_ASCEND_HOME, "tools", "ccec_compiler", "bin", "bisheng"),
    os.path.join(_ASCEND_HOME, "compiler", "ccec_compiler", "bin", "bisheng"),
]

_ASCENDC_INCLUDE = os.path.join(_ASCEND_HOME, "aarch64-linux", "ascendc", "include")
_ASC_INCLUDE = os.path.join(_ASCEND_HOME, "aarch64-linux", "asc", "include")


def _find_bisheng() -> str:
    """Find the bisheng compiler binary."""
    for p in _BISHENG_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "bisheng compiler not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in _BISHENG_CANDIDATES)
        + f"\nSet ASCEND_TOOLKIT_HOME environment variable."
    )


def _compile(source: str, options: list[str] | None,
             arch: str, core_type: str) -> Path:
    """Compile Ascend C source via bisheng, return path to kernel.o."""
    bisheng = _find_bisheng()
    build_dir = Path(tempfile.mkdtemp(prefix="asnumpy_"))
    src_file = build_dir / "kernel.cpp"
    src_file.write_text(source, encoding="utf-8")

    include_dirs = [
        _ASCENDC_INCLUDE,
        os.path.join(_ASCENDC_INCLUDE, "basic_api"),
        os.path.join(_ASCENDC_INCLUDE, "highlevel_api"),
        os.path.join(_ASCENDC_INCLUDE, "basic_api", "impl"),
        os.path.join(_ASCENDC_INCLUDE, "basic_api", "interface"),
    ]
    # CANN 9.x: additional aarch64-linux/asc paths for utility headers
    # (e.g., include/utils/std/tuple.h)
    asc_root = os.path.join(_ASCEND_HOME, "aarch64-linux", "asc")
    if os.path.isdir(asc_root):
        include_dirs.append(asc_root)
        include_dirs.append(os.path.join(asc_root, "include"))

    cmd = [bisheng]
    if options:
        cmd.extend(options)
    cmd += [
        f"--cce-soc-version={arch}",
        f"--cce-soc-core-type={core_type}",
        "--cce-aicore-lang",
        "--cce-aicore-arch=da-vinci",
        "--std=c++17", "-O2", "-fPIC", "-shared",
    ]
    for inc in include_dirs:
        cmd.append(f"-I{inc}")
    cmd += [str(src_file), "-o", str(build_dir / "kernel.o")]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"bisheng compilation failed (exit {result.returncode}):\n"
            f"{result.stderr}"
        )
    o_file = build_dir / "kernel.o"
    if not o_file.exists():
        raise RuntimeError("bisheng succeeded but kernel.o was not produced")
    return o_file


def _extract_elf(o_file: Path) -> bytes:
    """Extract .aicore_binary section (inner AI Core ELF) from bisheng .o."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".elf") as tmp:
        tmp_path = tmp.name
    try:
        result = subprocess.run(
            ["objcopy", "--dump-section", f".aicore_binary={tmp_path}",
             str(o_file)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"objcopy failed to extract .aicore_binary:\n{result.stderr}"
            )
        data = Path(tmp_path).read_bytes()
        if not data:
            raise RuntimeError(".aicore_binary section is empty")
        return data
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Symbol listing
# ---------------------------------------------------------------------------

_KERNEL_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+'
    r'(\w+)\s*\([^)]*\)'
)


def _find_kernels(source: str) -> list[str]:
    """Find kernel function names in Ascend C source."""
    return [m.group(1) for m in _KERNEL_RE.finditer(source)]


# ---------------------------------------------------------------------------
# SourceModule
# ---------------------------------------------------------------------------


class SourceModule:
    """JIT-compile an Ascend C kernel source string and expose callable functions.

    Analogous to :class:`pycuda.compiler.SourceModule`.

    Parameters
    ----------
    source:
        Ascend C kernel source code.
    options:
        Additional bisheng compiler options (e.g. ``["-O3"]``).
    arch:
        Target SoC version (default ``"Ascend910B4"``).
    core_type:
        AI Core type (default ``"VecCore"``).

    Examples
    --------
    >>> mod = SourceModule('''
    ...     extern "C" __global__ __aicore__ void my_add(
    ...         __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
    ...     { /* Ascend C kernel body */ }
    ... ''')
    >>> kernel = mod.get_function("my_add")
    >>> kernel(a_npu, b_npu, c_npu, 1024, grid=(8,))
    """

    def __init__(
        self,
        source: str,
        options: list[str] | None = None,
        arch: str = "Ascend910B4",
        core_type: str = "VecCore",
    ):
        self.source = source
        self._bin_handle: int | None = None
        self._kernels: list[str] = []

        # 1. Pre-check
        if 'extern "C"' not in source:
            warnings.warn(
                "Kernel source does not contain 'extern \"C\"'. "
                "Kernels may not be discoverable.",
                stacklevel=2,
            )

        # 2. Compile
        o_path = _compile(source, options, arch, core_type)

        # 3. Extract inner AI Core ELF from .aicore_binary section
        elf_data = _extract_elf(o_path)

        # 4. Register binary via RTS
        self._bin_handle = _rt.register_binary(elf_data)

        # 5. Discover and register kernel functions
        self._kernels = _find_kernels(source)
        for name in self._kernels:
            _rt.register_function(self._bin_handle, name)

    def get_function(self, name: str,
                     signature: list[str] | None = None) -> KernelFunction:
        """Return a callable :class:`KernelFunction` for the named kernel.

        Parameters
        ----------
        name:
            Kernel function name.
        signature:
            Optional explicit type signature, e.g. ``["float32*", "int32"]``.
        """
        if name not in self._kernels:
            raise ValueError(
                f"Kernel '{name}' not found. "
                f"Available: {list(self._kernels)}"
            )
        return KernelFunction(name, signature, bin_handle=self._bin_handle)

    def __del__(self):
        if self._bin_handle is not None:
            try:
                _rt.unregister_binary(self._bin_handle)
            except Exception:
                pass
