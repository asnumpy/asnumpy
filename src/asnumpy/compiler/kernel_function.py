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

"""KernelFunction and PreparedKernel -- callable wrappers for compiled Ascend C kernels."""

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


def _get_lib():
    """Lazy import of the C extension module (asnumpy._core.compiler).

    The module-level import is deferred so that pure-Python code paths
    (e.g. signature parsing, compilation) can be tested without a built
    C extension.
    """
    from .._core import compiler as cmod
    return cmod


def _rts_launch(kernel_name: str, block_dim: int, packed: list[bytes], stream) -> None:
    """Launch a kernel registered via ``rtDevBinaryRegister``."""
    from . import _rts_loader
    _rts_loader.launch_kernel(kernel_name, block_dim, packed, stream or 0)


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

_SCALAR_DTYPE_MAP: dict[str, type] = {
    "int": np.int32, "int32": np.int32, "int32_t": np.int32,
    "int64": np.int64, "int64_t": np.int64, "long long": np.int64,
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "bool": np.bool_,
}

_SIZEOF_MAP: dict[str, int] = {
    "int32": 4, "int64": 8,
    "float32": 4, "float64": 8, "float16": 2,
    "bool": 1,
}


# ---------------------------------------------------------------------------
# ArgSpec
# ---------------------------------------------------------------------------

@dataclass
class ArgSpec:
    """Describes a single kernel function parameter."""
    name: str           # parameter name from source
    arg_type: str       # "float32*", "int32", "int64", "float32", etc.
    is_pointer: bool    # True for __gm__ pointer parameters
    size_bytes: int     # sizeof in bytes (8 for pointer, 4 for int32, etc.)


# ---------------------------------------------------------------------------
# KernelFunction
# ---------------------------------------------------------------------------

class KernelFunction:
    """A callable representing a single compiled Ascend C kernel function.

    Created by :meth:`SourceModule.get_function`. Calling the instance launches
    the kernel on the NPU with the given arguments.

    Parameters
    ----------
    name:
        The kernel function name (as declared in source).
    func_handle:
        Opaque function handle from the C++ layer.
    arg_specs:
        Parsed parameter specifications for argument marshalling.
    """

    def __init__(
        self,
        name: str,
        func_handle: int,
        arg_specs: list[ArgSpec],
        use_rts: bool = False,
    ):
        self.name = name
        self._func_handle = func_handle
        self._arg_specs = arg_specs
        self._use_rts = use_rts

    def __repr__(self) -> str:
        spec_str = ", ".join(
            f"{s.name}: {s.arg_type}" for s in self._arg_specs
        )
        return f"KernelFunction({self.name}({spec_str}))"

    def __call__(
        self,
        *args,
        grid: tuple[int, ...] | None = None,
        block_shape=None,  # ignored, for PyCUDA API compatibility
        stream=None,
    ) -> None:
        """Launch the kernel on the NPU.

        Parameters
        ----------
        *args:
            Positional arguments matching the kernel parameter list.
            ``ndarray`` for ``__gm__`` pointer parameters, Python scalars otherwise.
        grid:
            Number of AI Cores to use, e.g. ``(8,)`` for 8 cores.
            Defaults to ``(1,)`` (single core).
        block_shape:
            Ignored. Ascend C uses SPMD, not CUDA-style threads.
        stream:
            ACL stream handle or ``None`` for the default (synchronous) stream.
        """
        if grid is None:
            grid = (1,)
        block_dim = grid[0]
        if len(grid) > 1:
            raise NotImplementedError(
                "Only 1D grids are currently supported"
            )

        packed = self._marshal_args(args)
        if self._use_rts:
            _rts_launch(self.name, block_dim, packed, stream or 0)
        else:
            _get_lib().launch_kernel(
                self._func_handle, block_dim, packed, stream or 0
            )

    def prepare(self) -> "PreparedKernel":
        """Return a :class:`PreparedKernel` that supports timed execution."""
        return PreparedKernel(self)

    def _marshal_args(self, args: tuple) -> list[bytes]:
        """Pack Python arguments into raw byte buffers for the kernel.

        - Pointer args (``__gm__ T*``): pack the 8-byte device address.
        - Scalar args: convert to the correct numpy dtype and call ``.tobytes()``.
        """
        n_expected = len(self._arg_specs)
        if len(args) != n_expected:
            raise TypeError(
                f"{self.name} expects {n_expected} arguments, got {len(args)}"
            )

        packed = []
        for arg, spec in zip(args, self._arg_specs):
            if spec.is_pointer:
                # Extract NPU device pointer from ndarray
                device_ptr = getattr(arg, "device_address", None)
                if device_ptr is None:
                    raise TypeError(
                        f"Argument '{spec.name}': expected an ndarray "
                        f"(has device_address), got {type(arg).__name__}"
                    )
                packed.append(struct.pack("<Q", device_ptr))
            else:
                # Scalar: convert to correct numpy type, then to bytes
                np_dtype = _SCALAR_DTYPE_MAP.get(spec.arg_type, np.int32)
                scalar = np_dtype(arg)
                packed.append(scalar.tobytes())
        return packed


# ---------------------------------------------------------------------------
# PreparedKernel
# ---------------------------------------------------------------------------

class PreparedKernel:
    """A kernel callable with baked-in arguments and performance timing.

    Create via :meth:`KernelFunction.prepare`.  Supports repeated launches
    with ACL event-based timing.

    Attributes
    ----------
    time:
        Elapsed time of the most recent launch in milliseconds.
    """

    def __init__(self, kernel: KernelFunction):
        self._kernel = kernel
        self._func_handle = kernel._func_handle
        self._start_event = _get_lib().create_event()
        self._end_event = _get_lib().create_event()
        self._stream = _get_lib().create_stream()
        self._last_time_ms = 0.0

    @property
    def time(self) -> float:
        """Elapsed time of the last invocation (milliseconds)."""
        return self._last_time_ms

    def __call__(self, *args, stream=None) -> None:
        """Launch with pre-configured timing."""
        packed = self._kernel._marshal_args(args)
        block_dim = 1  # default, caller may override via grid

        # Record start event
        _get_lib().record_event(self._start_event, self._stream)

        # Launch
        if self._kernel._use_rts:
            _rts_launch(self._kernel.name, block_dim, packed,
                        stream or self._stream)
        else:
            _get_lib().launch_kernel(
                self._func_handle, block_dim, packed,
                stream or self._stream,
            )

        # Record end event
        _get_lib().record_event(self._end_event, self._stream)

        # Synchronize and measure
        _get_lib().synchronize_event(self._end_event)
        self._last_time_ms = _get_lib().elapsed_time_between(
            self._start_event, self._end_event
        )

    def __del__(self):
        try:
            _get_lib().destroy_event(self._start_event)
        except Exception:
            pass
        try:
            _get_lib().destroy_event(self._end_event)
        except Exception:
            pass
        try:
            _get_lib().destroy_stream(self._stream)
        except Exception:
            pass
