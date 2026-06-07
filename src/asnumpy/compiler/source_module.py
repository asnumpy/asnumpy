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

"""SourceModule -- JIT-compile and launch Ascend C kernels from Python source strings.

Analogous to ``pycuda.compiler.SourceModule``.
"""

import atexit
import re
import subprocess
import tempfile
import warnings
from pathlib import Path

from . import bisheng_compiler, cache
from .kernel_function import ArgSpec, KernelFunction


def _get_lib():
    """Lazy import of the C extension module (asnumpy._core.compiler).

    The module-level import is deferred so that pure-Python code paths
    (e.g. signature parsing, compilation) can be tested without a built
    C extension.
    """
    from .._core import compiler as cmod
    return cmod

# ---------------------------------------------------------------------------
# Regex patterns for signature parsing
# ---------------------------------------------------------------------------

_SIGNATURE_RE = re.compile(
    r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+'
    r'(\w+)\s*\(([^)]*)\)'
)

_PTR_PARAM_RE = re.compile(
    r'__gm__\s+(const\s+)?(\w+)\s*\*\s*(?:__restrict__\s+)?(\w+)'
)

_SCALAR_PARAM_RE = re.compile(
    r'\b(int|int32_t|int64_t|long\s+long|float|double|bool|half|short)\s+(\w+)'
)

_C_TYPE_MAP: dict[str, tuple[str, int]] = {
    "int": ("int32", 4), "int32_t": ("int32", 4),
    "int64_t": ("int64", 8), "long long": ("int64", 8),
    "float": ("float32", 4),
    "double": ("float64", 8),
    "bool": ("bool", 1),
    "half": ("float16", 2),
    "short": ("int16", 2),
    # convenience aliases
    "int32": ("int32", 4),
    "int64": ("int64", 8),
    "float32": ("float32", 4),
    "float64": ("float64", 8),
}

# ---------------------------------------------------------------------------
# atexit cleanup registry
# ---------------------------------------------------------------------------

_atexit_registry: list["SourceModule"] = []


@atexit.register
def _cleanup_all_sources() -> None:
    """Release all binary handles before CANN finalize.

    Registered via atexit; since SourceModule instances are created after
    asnumpy/__init__.py's own atexit callback (which calls aclFinalize),
    this callback runs FIRST (LIFO order), unloading binaries while CANN
    is still alive.
    """
    for mod in _atexit_registry:
        try:
            mod.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Signature parsing
# ---------------------------------------------------------------------------

def _parse_kernel_signature(source: str, kernel_name: str) -> list[ArgSpec]:
    """Parse the parameter list of a specific kernel function from source code.

    Parameters
    ----------
    source:
        Ascend C kernel source string.
    kernel_name:
        The kernel function name to look for.

    Returns
    -------
    list of ArgSpec
        Parsed parameter specifications.

    Raises
    ------
    ValueError
        If the kernel is not found or parameter parsing fails.
    """
    for match in _SIGNATURE_RE.finditer(source):
        if match.group(1) == kernel_name:
            params_str = match.group(2).strip()
            if not params_str:
                return []
            specs = []
            for param in params_str.split(","):
                param = param.strip()
                if not param:
                    continue

                # Try pointer parameter first (__gm__ float* name)
                ptr_match = _PTR_PARAM_RE.match(param)
                if ptr_match:
                    type_name = ptr_match.group(2)
                    param_name = ptr_match.group(3)
                    specs.append(ArgSpec(
                        name=param_name,
                        arg_type=f"{type_name}*",
                        is_pointer=True,
                        size_bytes=8,
                    ))
                    continue

                # Try scalar parameter
                scalar_match = _SCALAR_PARAM_RE.match(param)
                if scalar_match:
                    c_type = scalar_match.group(1)
                    param_name = scalar_match.group(2)
                    type_info = _C_TYPE_MAP.get(c_type, ("int32", 4))
                    specs.append(ArgSpec(
                        name=param_name,
                        arg_type=type_info[0],
                        is_pointer=False,
                        size_bytes=type_info[1],
                    ))
                    continue

                raise ValueError(
                    f"Cannot parse kernel parameter: '{param}' "
                    f"in kernel '{kernel_name}'"
                )
            return specs

    raise ValueError(f"Kernel '{kernel_name}' not found in source")


def _manual_signature_to_arg_specs(signature: list[str]) -> list[ArgSpec]:
    """Convert a user-provided signature list to ArgSpec objects.

    Signature entries look like ``"float32*"`` (pointer) or ``"int32"`` (scalar).
    """
    specs = []
    for i, entry in enumerate(signature):
        entry = entry.strip()
        if entry.endswith("*"):
            type_name = entry[:-1]
            specs.append(ArgSpec(
                name=f"arg{i}",
                arg_type=entry,
                is_pointer=True,
                size_bytes=8,
            ))
        else:
            type_info = _C_TYPE_MAP.get(entry, (entry, 4))
            specs.append(ArgSpec(
                name=f"arg{i}",
                arg_type=type_info[0],
                is_pointer=False,
                size_bytes=type_info[1],
            ))
    return specs


# ---------------------------------------------------------------------------
# Kernel symbol listing
# ---------------------------------------------------------------------------

def _list_kernel_symbols(o_file: str) -> list[str]:
    """Extract kernel function names from a compiled .o file.

    Uses ``objdump -t`` to parse the symbol table and find global text symbols
    (``extern "C"`` kernel entry points).
    """
    try:
        result = subprocess.run(
            ["objdump", "-t", o_file],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try nm
        try:
            result = subprocess.run(
                ["nm", o_file],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []

    symbols = []
    for line in result.stdout.splitlines():
        parts = line.split()
        # objdump format: address flags section ... name
        # nm format:      address type name
        if len(parts) >= 3:
            # objdump: "g" flag and ".text" section indicates global text symbol
            if len(parts) >= 5 and parts[1] == "g" and parts[3] == ".text":
                name = parts[-1]
            # nm: "T" or "t" type indicates text (code) symbol
            elif parts[1] in ("T", "t") and len(parts) == 3:
                name = parts[-1]
            else:
                continue
            if name and not name.startswith(".") and not name.startswith("_"):
                symbols.append(name)
    return list(set(symbols))  # deduplicate


# ---------------------------------------------------------------------------
# Source pre-check
# ---------------------------------------------------------------------------

def _check_extern_c(source: str) -> None:
    """Warn if the source does not contain ``extern "C"``."""
    if 'extern "C"' not in source:
        warnings.warn(
            "Kernel source does not contain 'extern \"C\"'. "
            "Kernel functions may not be found by aclrtBinaryGetFunction.",
            stacklevel=3,
        )


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
    include_dirs:
        User-specified additional include directories.
    cache_dir:
        Cache directory path. Defaults to ``~/.asnumpy/cache/``.
    disable_cache:
        If ``True``, skip caching and always recompile.
    keep:
        If ``True``, keep intermediate build artifacts.
    verbose:
        If ``True``, print compilation commands and progress.
    soc_version:
        Target SoC version (e.g. ``"Ascend910B1"``).
    core_type:
        AI Core type (``"VecCore"``, ``"CubeCore"``, or ``"AICore"``).
    compiler:
        Path to bisheng compiler. ``None`` to auto-detect from ``ASCEND_TOOLKIT_HOME``.

    Examples
    --------
    >>> mod = SourceModule('''
    ...     #include "kernel_operator.h"
    ...     using namespace AscendC;
    ...     extern "C" __global__ __aicore__ void my_add(
    ...         __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
    ...     { /* ... */ }
    ... ''')
    >>> kernel = mod.get_function("my_add")
    >>> kernel(a_npu, b_npu, c_npu, 1024, grid=(8,))
    """

    def __init__(
        self,
        source: str,
        options: list[str] | None = None,
        include_dirs: list[str] | None = None,
        cache_dir: str | None = None,
        disable_cache: bool = False,
        keep: bool = False,
        verbose: bool = False,
        soc_version: str = "Ascend910B4",
        core_type: str = "VecCore",
        compiler: str | None = None,
    ):
        self.source = source
        self._options = options or []
        self._include_dirs = include_dirs
        self._keep = keep
        self._verbose = verbose
        self._soc_version = soc_version
        self._core_type = core_type

        self._bin_handle: int | None = None
        self._functions: dict[str, int] = {}     # name -> func_handle
        self._kernel_names: list[str] = []
        self._closed = False
        self._build_dir: Path | None = None

        # 1. Check for extern "C"
        _check_extern_c(source)

        # 2. Compute cache key
        compiler_ver = cache.get_compiler_version()
        self._cache_key = cache.compute_cache_key(
            source, self._options, soc_version, compiler_ver, include_dirs
        )

        # 3. Compile (or load from cache)
        o_path: str | None = None

        if not disable_cache:
            cached = cache.cache_hit(self._cache_key)
            if cached is not None:
                o_path = str(cached)
                if verbose:
                    print(f"[SourceModule] Cache hit: {cached}")

        if o_path is None:
            o_path = self._compile()

            if not disable_cache and self._build_dir is not None:
                cache.store_in_cache(self._cache_key, self._build_dir)

        # 4. Load binary via RTS path (rtDevBinaryRegister with inner ELF).
        #    The inner ELF is extracted from the .aicore_binary section of the
        #    bisheng-compiled .o.  We use RT_DEV_BINARY_MAGIC_ELF_AIVEC
        #    (0x41415246) which matches the VecCore output.
        from . import _rts_loader

        aicore_elf = bisheng_compiler.extract_aicore_elf(Path(o_path))
        self._bin_handle = _rts_loader.register_binary(aicore_elf)

        # 5. Discover kernel symbols and register them
        self._kernel_names = _list_kernel_symbols(o_path)
        self._functions: dict[str, int] = {}     # name -> bin_handle
        for name in self._kernel_names:
            try:
                _rts_loader.register_function(self._bin_handle, name)
            except Exception as e:
                if verbose:
                    print(f"[SourceModule] Warning: failed to register "
                          f"'{name}': {e}")

        # 6. Register for atexit cleanup
        _atexit_registry.append(self)

    # -- public API ----------------------------------------------------------

    def get_function(
        self, name: str, signature: list[str] | None = None
    ) -> KernelFunction:
        """Get a callable :class:`KernelFunction` for the named kernel.

        Parameters
        ----------
        name:
            Kernel function name.
        signature:
            Optional explicit signature list (e.g. ``["float32*", "int32"]``).
            If ``None``, the signature is parsed from the source code.
        """
        if name not in self._kernel_names:
            available = list(self._kernel_names)
            raise ValueError(
                f"Kernel '{name}' not found. Available: {available}"
            )

        if signature is not None:
            arg_specs = _manual_signature_to_arg_specs(signature)
        else:
            arg_specs = _parse_kernel_signature(self.source, name)

        return KernelFunction(name, 0, arg_specs, use_rts=True)

    def list_functions(self) -> list[str]:
        """Return the list of kernel function names in this module."""
        return list(self._kernel_names)

    def close(self) -> None:
        """Release the binary handle. Safe to call multiple times."""
        if not self._closed and self._bin_handle is not None:
            from . import _rts_loader
            _rts_loader.unregister_binary(self._bin_handle)
            self._bin_handle = None
            self._functions.clear()
            self._closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SourceModule":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- internal ------------------------------------------------------------

    def _compile(self) -> str:
        """Compile source via bisheng, return path to .o file."""
        if self._keep:
            build_dir = Path(tempfile.mkdtemp(prefix="asnumpy_src_"))
        else:
            build_dir = Path(tempfile.mkdtemp(prefix="asnumpy_compile_"))

        self._build_dir = build_dir

        # Write source to build dir for cache storage
        (build_dir / "source.cpp").write_text(self.source, encoding="utf-8")

        o_file = bisheng_compiler.compile_kernel(
            self.source,
            build_dir,
            options=self._options,
            include_dirs=self._include_dirs,
            soc_version=self._soc_version,
            core_type=self._core_type,
            verbose=self._verbose,
        )
        return str(o_file)
