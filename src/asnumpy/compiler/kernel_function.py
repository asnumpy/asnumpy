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

"""KernelFunction — a callable wrapper for a compiled Ascend C kernel."""

import struct

import numpy as np

from . import _rt

# ---------------------------------------------------------------------------
# Scalar type → numpy dtype mapping (for marshalling)
# ---------------------------------------------------------------------------
_SCALAR_DTYPE: dict[str, type] = {
    "int": np.int32, "int32": np.int32, "int32_t": np.int32,
    "int64": np.int64, "int64_t": np.int64, "long long": np.int64,
    "float": np.float32, "float32": np.float32,
    "double": np.float64, "float64": np.float64,
    "bool": np.bool_,
}


class KernelFunction:
    """A callable representing a single compiled Ascend C kernel function.

    Created by :meth:`SourceModule.get_function`. Calling the instance launches
    the kernel on the NPU.

    Parameters
    ----------
    name:
        Kernel function name (as declared in ``extern "C"`` source).
    signature:
        Explicit parameter type list, e.g. ``["float32*", "float32*", "int32"]``.
        ``T*`` entries are treated as pointer (__gm__) parameters and expect an
        ``ndarray`` argument. Other entries are scalar types.
    bin_handle:
        Opaque binary handle from SourceModule (ACL path).
    """

    def __init__(self, name: str, signature: list[str] | None = None,
                 bin_handle: int | None = None):
        self.name = name
        self._signature = signature or []
        self._bin_handle = bin_handle

    def __repr__(self) -> str:
        sig = ", ".join(self._signature) if self._signature else "?"
        return f"KernelFunction({self.name}({sig}))"

    def __call__(self, *args, grid: tuple[int, ...] | None = None,
                 stream=None) -> None:
        """Launch the kernel on the NPU.

        Parameters
        ----------
        *args:
            Positional arguments matching the kernel parameter list.
            ``ndarray`` for pointer (``__gm__ T*``) parameters,
            Python scalars for value parameters.
        grid:
            AI Core count, e.g. ``(8,)``. Default ``(1,)``.
        stream:
            Reserved for future use (currently ignored).
        """
        block_dim = 1 if grid is None else grid[0]
        packed = self._pack_args(args)
        _rt.launch_kernel(self.name, block_dim, packed, stream or 0)

    def _pack_args(self, args: tuple) -> list[bytes]:
        """Pack Python arguments into raw byte buffers for the kernel.

        - Pointer args: pack 8-byte device address (little-endian).
        - Scalar args: convert to numpy dtype then ``.tobytes()``.
        """
        if self._signature and len(args) != len(self._signature):
            raise TypeError(
                f"{self.name} expects {len(self._signature)} arguments, "
                f"got {len(args)}"
            )
        packed = []
        for i, arg in enumerate(args):
            sig = self._signature[i] if i < len(self._signature) else None
            if sig and sig.endswith("*"):
                # Pointer parameter: extract device address from ndarray
                ptr = getattr(arg, "device_address", None)
                if ptr is None:
                    raise TypeError(
                        f"Argument {i}: expected ndarray (pointer param), "
                        f"got {type(arg).__name__}"
                    )
                packed.append(struct.pack("<Q", ptr))
            else:
                # Scalar parameter
                dtype_name = sig.rstrip("*") if sig else "int32"
                np_dtype = _SCALAR_DTYPE.get(dtype_name, np.int32)
                packed.append(np_dtype(arg).tobytes())
        return packed
