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

"""Invoke the bisheng compiler to compile Ascend C kernel source into a .o file."""

import os
import subprocess
import tempfile
from pathlib import Path


class CompileError(Exception):
    """Raised when bisheng compilation fails. Contains the full stderr output."""

    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def _get_ascend_toolkit_home() -> str:
    """Get the CANN toolkit installation path."""
    path = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if not path:
        path = os.environ.get("ASCEND_HOME_PATH", "")
    if not path:
        path = "/usr/local/Ascend/ascend-toolkit/latest"
    return path


def _get_bisheng_path() -> str:
    """Return the full path to the bisheng compiler."""
    ascend_home = _get_ascend_toolkit_home()
    bisheng = os.path.join(ascend_home, "compiler", "ccec_compiler", "bin", "bisheng")
    if not os.path.isfile(bisheng):
        raise FileNotFoundError(
            f"bisheng compiler not found at {bisheng}. "
            f"Set ASCEND_TOOLKIT_HOME environment variable."
        )
    return bisheng


def _get_ascendc_include_paths() -> list[str]:
    """Return the list of Ascend C include directories needed for kernel compilation."""
    ascend_home = _get_ascend_toolkit_home()
    base = os.path.join(ascend_home, "aarch64-linux", "ascendc", "include")
    return [
        base,
        os.path.join(base, "basic_api"),
        os.path.join(base, "highlevel_api"),
        os.path.join(base, "basic_api", "impl"),
        os.path.join(base, "basic_api", "inner_interface"),
        os.path.join(base, "basic_api", "interface"),
    ]


def compile_kernel(
    source: str,
    output_dir: Path,
    *,
    options: list[str] | None = None,
    include_dirs: list[str] | None = None,
    soc_version: str = "Ascend910B1",
    core_type: str = "VecCore",
    verbose: bool = False,
) -> Path:
    """Compile Ascend C kernel source code into a .o file.

    Args:
        source: Ascend C kernel source code string.
        output_dir: Directory to write the compiled .o and intermediate files.
        options: Additional bisheng compiler options (e.g. ["-O3"]).
        include_dirs: Additional user-specified include directories.
        soc_version: Target SoC version (e.g. "Ascend910B1").
        core_type: AI Core type ("VecCore", "CubeCore", or "AICore").
        verbose: If True, print compilation command and output.

    Returns:
        Path to the compiled .so file (linked shared object).

    Raises:
        CompileError: If compilation fails.
        FileNotFoundError: If the bisheng compiler is not found.

    Note:
        The kernel is compiled directly to a shared object (.so) using ``-shared``
        and ``-fPIC``. This format is required by ``aclrtBinaryLoadFromFile``
        (available in CANN 8.5+).
    """
    bisheng = _get_bisheng_path()
    ascendc_includes = _get_ascendc_include_paths()

    # Write source to a temporary .cpp file
    source_file = output_dir / "kernel.cpp"
    source_file.write_text(source, encoding="utf-8")

    # Assemble include paths: user-specified first, then Ascend C system paths
    all_includes = list(include_dirs or [])
    all_includes.extend(ascendc_includes)

    # Build command
    cmd = [
        bisheng,
        f"--cce-soc-version={soc_version}",
        f"--cce-soc-core-type={core_type}",
        "--cce-aicore-lang",
        "--std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(source_file),
        "-o", str(output_dir / "kernel.so"),
    ]

    # Insert user options before the standard ones
    if options:
        cmd[2:2] = options

    # Insert include paths after options
    include_args = []
    for inc in all_includes:
        include_args.extend([f"-I{inc}"])
    # Insert include dirs after the standard flags (after -fPIC)
    pos = cmd.index("-fPIC") + 1
    cmd[pos:pos] = include_args

    if verbose:
        print(f"[bisheng] {' '.join(cmd)}")

    # Run bisheng
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # Write compilation log
    log_file = output_dir / "compile.log"
    log_file.write_text(result.stderr + "\n" + result.stdout, encoding="utf-8")

    if result.returncode != 0:
        raise CompileError(
            f"bisheng compilation failed (exit code {result.returncode}):\n{result.stderr}",
            stderr=result.stderr,
        )

    so_file = output_dir / "kernel.so"
    if not so_file.exists():
        raise CompileError(
            "bisheng compilation succeeded but .so file was not produced",
            stderr=result.stderr,
        )

    return so_file
