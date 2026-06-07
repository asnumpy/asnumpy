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
    """Return the full path to the bisheng compiler.

    Probes multiple known locations to support CANN 7.x through 9.x:
      - CANN 9.x: ``tools/bisheng_compiler/bin/bisheng``
      - CANN 9.x (alt): ``tools/ccec_compiler/bin/bisheng``
      - CANN 7.x / 8.x: ``compiler/ccec_compiler/bin/bisheng``
    """
    ascend_home = _get_ascend_toolkit_home()
    candidates = [
        os.path.join(ascend_home, "tools", "bisheng_compiler", "bin", "bisheng"),
        os.path.join(ascend_home, "tools", "ccec_compiler", "bin", "bisheng"),
        os.path.join(ascend_home, "compiler", "ccec_compiler", "bin", "bisheng"),
    ]
    for bisheng in candidates:
        if os.path.isfile(bisheng):
            return bisheng
    raise FileNotFoundError(
        f"bisheng compiler not found. Checked:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + f"\nSet ASCEND_TOOLKIT_HOME environment variable."
    )


def _get_ascendc_include_paths() -> list[str]:
    """Return the list of Ascend C include directories needed for kernel compilation.

    CANN 9.x adds two ``aarch64-linux/asc/`` paths that supply utility headers
    (e.g. ``include/utils/std/tuple.h``) referenced by the public Ascend C
    headers.  These directories are safe to add on CANN 7.x / 8.x as well
    (they simply won't exist yet).
    """
    ascend_home = _get_ascend_toolkit_home()
    base = os.path.join(ascend_home, "aarch64-linux", "ascendc", "include")
    asc_root = os.path.join(ascend_home, "aarch64-linux", "asc")
    asc_include = os.path.join(asc_root, "include")
    return [
        asc_root,
        asc_include,
        base,
        os.path.join(base, "basic_api"),
        os.path.join(base, "highlevel_api"),
        os.path.join(base, "basic_api", "impl"),
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
        Path to the compiled .o file (relocatable ELF object).

    Raises:
        CompileError: If compilation fails.
        FileNotFoundError: If the bisheng compiler is not found.

    Note:
        The kernel is compiled with ``-shared`` (required to resolve Ascend C
        runtime symbols) but written as ``kernel.o``. The ``--cce-aicore-arch=da-vinci``
        flag generates AI Core compatible ELF section layout.
        Whether this format is accepted by ``aclrtBinaryLoadFromFile`` in CANN 8.2
        remains to be verified on hardware.
    """
    bisheng = _get_bisheng_path()
    ascendc_includes = _get_ascendc_include_paths()

    # Write source to a temporary .cpp file
    source_file = output_dir / "kernel.cpp"
    source_file.write_text(source, encoding="utf-8")

    # Assemble include paths: user-specified first, then Ascend C system paths
    all_includes = list(include_dirs or [])
    all_includes.extend(ascendc_includes)

    # Build command — produce .o for CANN 8.2 compatibility.
    # NOTE: -shared is required; Ascend C kernels reference runtime symbols
    # (rtLaunch, rtFunctionRegister, etc.) that must be resolved at link time.
    # Removing -shared causes lld to produce an executable (needs main).
    # --cce-aicore-arch=da-vinci generates proper AI Core ELF sections.
    cmd = [
        bisheng,
        f"--cce-soc-version={soc_version}",
        f"--cce-soc-core-type={core_type}",
        "--cce-aicore-lang",
        "--cce-aicore-arch=da-vinci",
        "--std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(source_file),
        "-o", str(output_dir / "kernel.o"),
    ]

    # Insert user options before the standard ones
    if options:
        cmd[2:2] = options

    # Insert include paths after options
    include_args = []
    for inc in all_includes:
        include_args.extend([f"-I{inc}"])
    # Insert include dirs after -shared (the last standard flag before positional args)
    pos = cmd.index("-shared") + 1
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

    o_file = output_dir / "kernel.o"
    if not o_file.exists():
        raise CompileError(
            "bisheng compilation succeeded but .o file was not produced",
            stderr=result.stderr,
        )

    return o_file


def extract_aicore_elf(o_file: Path) -> bytes:
    """Extract the inner AI Core ELF from a bisheng-compiled .o file.

    Bisheng wraps the device binary inside a ``.aicore_binary`` section of the
    outer host-side ELF.  This function shells out to ``objcopy`` to dump
    that section, returning the raw device ELF bytes suitable for registration
    via ``rtDevBinaryRegister``.
    """
    import subprocess
    import tempfile

    # objcopy --dump-section is the reliable way to extract a section to a file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".elf") as tmp:
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["objcopy", "--dump-section",
             f".aicore_binary={tmp_path}", str(o_file)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CompileError(
                f"objcopy failed to extract .aicore_binary section:\n{result.stderr}",
                stderr=result.stderr,
            )
        data = Path(tmp_path).read_bytes()
        if not data:
            raise CompileError(
                ".aicore_binary section is empty — the bisheng output may have "
                "changed format. Check that --cce-aicore-lang and "
                "--cce-soc-core-type are set correctly."
            )
        return data
    finally:
        Path(tmp_path).unlink(missing_ok=True)
