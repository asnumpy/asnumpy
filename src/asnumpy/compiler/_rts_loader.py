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

"""RTS (Runtime Service) loader for CANN 9.x.

Uses the ``rtDevBinaryRegister`` + ``rtFunctionRegister`` + ``rtKernelLaunch``
API via ctypes to load and launch Ascend C kernels compiled by bisheng.

This module exists because the C++ extension's ``launch_kernel_rts`` hits
internal runtime errors (507000) when calling ``rtKernelLaunch`` through the
compile-time-linked libruntime.  Using ctypes to load libruntime provides
a self-contained environment where registration and launch share state.

**Key implementation detail:** CANN 9.x ``rtKernelLaunch`` compares the
``stubFunc`` **pointer** (not the string content) against the ``stubFunc``
pointer stored by ``rtFunctionRegister``.  If ``register_function`` and
``launch_kernel`` use different bytes objects (even with identical content),
``rtKernelLaunch`` returns rtError=507000.  We work around this by storing
the ``bytes`` object in ``_kernel_name_refs`` during ``register_function``
and reusing it in ``launch_kernel``.

The inner ELF from the ``.aicore_binary`` section of a bisheng-compiled .o
file is registered with ``RT_DEV_BINARY_MAGIC_ELF_AIVEC`` (0x41415246) which
matches ``--cce-soc-core-type=VecCore`` output.
"""

import ctypes
import os
import struct
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246  # "FRAA" in little-endian
FUNC_MODE_NORMAL = 0

# ---------------------------------------------------------------------------
# Structs
# ---------------------------------------------------------------------------


class RtDevBinary(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_uint64),
    ]


# ---------------------------------------------------------------------------
# Library loading (lazy, once per process)
# ---------------------------------------------------------------------------

_librt: Optional[ctypes.CDLL] = None
_libacl: Optional[ctypes.CDLL] = None

# Keep registered binary data alive — some CANN runtime versions access
# the data pointer lazily after rtDevBinaryRegister returns.
_registry: dict[int, object] = {}

# Keep kernel name bytes alive — CANN 9.x rtKernelLaunch compares the stubFunc
# POINTER (not the string content) against the one stored by rtFunctionRegister.
# If register_function and launch_kernel use different bytes objects, even with
# the same content, rtKernelLaunch returns rtError=507000.
_kernel_name_refs: dict[str, bytes] = {}


def _get_librt() -> ctypes.CDLL:
    """Return the ctypes handle for libruntime.so (lazy singleton)."""
    global _librt
    if _librt is None:
        ah = os.environ.get(
            "ASCEND_TOOLKIT_HOME",
            os.environ.get("ASCEND_HOME_PATH",
                           "/usr/local/Ascend/ascend-toolkit/latest"),
        )
        lib_path = f"{ah}/lib64/libruntime.so"
        _librt = ctypes.CDLL(lib_path)
        _librt.rtDevBinaryRegister.argtypes = [
            ctypes.POINTER(RtDevBinary), ctypes.POINTER(ctypes.c_void_p),
        ]
        _librt.rtDevBinaryRegister.restype = ctypes.c_int
        _librt.rtFunctionRegister.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p,
            ctypes.c_void_p, ctypes.c_uint32,
        ]
        _librt.rtFunctionRegister.restype = ctypes.c_int
        _librt.rtKernelLaunch.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
        ]
        _librt.rtKernelLaunch.restype = ctypes.c_int
        _librt.rtDevBinaryUnRegister.argtypes = [ctypes.c_void_p]
        _librt.rtDevBinaryUnRegister.restype = ctypes.c_int
    return _librt


def _get_libacl() -> ctypes.CDLL:
    """Return the ctypes handle for libascendcl.so (lazy singleton)."""
    global _libacl
    if _libacl is None:
        ah = os.environ.get(
            "ASCEND_TOOLKIT_HOME",
            os.environ.get("ASCEND_HOME_PATH",
                           "/usr/local/Ascend/ascend-toolkit/latest"),
        )
        _libacl = ctypes.CDLL(f"{ah}/lib64/libascendcl.so")
        _libacl.aclrtCreateStream.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        _libacl.aclrtCreateStream.restype = ctypes.c_int
        _libacl.aclrtDestroyStream.argtypes = [ctypes.c_void_p]
        _libacl.aclrtDestroyStream.restype = ctypes.c_int
        _libacl.aclrtSynchronizeStream.argtypes = [ctypes.c_void_p]
        _libacl.aclrtSynchronizeStream.restype = ctypes.c_int
        _libacl.aclrtSynchronizeDevice.argtypes = []
        _libacl.aclrtSynchronizeDevice.restype = ctypes.c_int
    return _libacl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_binary(data: bytes) -> int:
    """Register a raw kernel binary (inner ELF from .aicore_binary section).

    Returns an opaque integer handle.
    """
    librt = _get_librt()
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    dev_bin = RtDevBinary(
        magic=RT_DEV_BINARY_MAGIC_ELF_AIVEC,
        version=0,
        data=ctypes.cast(buf, ctypes.c_void_p),
        length=len(data),
    )
    handle = ctypes.c_void_p()
    ret = librt.rtDevBinaryRegister(ctypes.byref(dev_bin), ctypes.byref(handle))
    if ret != 0:
        raise RuntimeError(f"rtDevBinaryRegister failed: rtError={ret}")
    h = handle.value  # type: ignore[return-value]
    # Keep the buffer alive — CANN runtime may access data lazily
    _registry[h] = buf
    return h


def register_function(bin_handle: int, kernel_name: str) -> None:
    """Register a kernel function by name in the binary.

    The encoded name bytes are kept alive in ``_kernel_name_refs`` so that
    later ``launch_kernel`` calls can reuse the **same** pointer.  CANN 9.x
    ``rtKernelLaunch`` compares the ``stubFunc`` pointer (not the string
    content), so without this the launch receives rtError=507000.
    """
    librt = _get_librt()
    name = kernel_name.encode()
    _kernel_name_refs[kernel_name] = name  # keep alive for pointer stability
    ret = librt.rtFunctionRegister(
        ctypes.c_void_p(bin_handle),
        name,  # stubFunc
        name,  # stubName
        name,  # kernelInfoExt
        FUNC_MODE_NORMAL,
    )
    if ret != 0:
        raise RuntimeError(
            f"rtFunctionRegister failed for '{kernel_name}': rtError={ret}"
        )


def unregister_binary(bin_handle: int) -> None:
    """Unregister a kernel binary."""
    librt = _get_librt()
    ret = librt.rtDevBinaryUnRegister(ctypes.c_void_p(bin_handle))
    if ret != 0:
        # Non-fatal warning
        import sys
        print(
            f"[asnumpy] rtDevBinaryUnRegister warning: rtError={ret}",
            file=sys.stderr,
        )
    _registry.pop(bin_handle, None)
    # Note: _kernel_name_refs entries are intentionally kept — they hold
    # bytes objects that may still be referenced by registered binaries.


def launch_kernel(
    kernel_name: str,
    block_dim: int,
    packed_args: list[bytes],
    stream: int = 0,
) -> None:
    """Launch a registered kernel via ``rtKernelLaunch``.

    Parameters
    ----------
    kernel_name:
        The name used when calling ``register_function``.
    block_dim:
        Number of AI Core blocks (grid dimension).
    packed_args:
        List of ``bytes`` objects, one per kernel parameter.
    stream:
        ACL stream handle or 0 for the default stream.
    """
    librt = _get_librt()
    libacl = _get_libacl()

    # Flatten args
    total = sum(len(a) for a in packed_args)
    flat = bytearray(total)
    offset = 0
    for a in packed_args:
        flat[offset:offset + len(a)] = a
        offset += len(a)
    args_buf = (ctypes.c_ubyte * total).from_buffer_copy(bytes(flat))

    # Reuse the EXACT bytes object from register_function — CANN 9.x
    # rtKernelLaunch compares the stubFunc pointer, not the string content.
    name = _kernel_name_refs.get(kernel_name)
    if name is None:
        # Fallback: encode here (works if register_function reused the same
        # bytes object from a previous call, e.g. via SourceModule cache).
        name = kernel_name.encode()
        _kernel_name_refs[kernel_name] = name
    ret = librt.rtKernelLaunch(name, block_dim, args_buf, total, None,
                                ctypes.c_void_p(stream))

    # Synchronize to surface kernel errors
    if stream != 0:
        sync_ret = libacl.aclrtSynchronizeStream(ctypes.c_void_p(stream))
    else:
        sync_ret = libacl.aclrtSynchronizeDevice()

    if ret != 0 or sync_ret != 0:
        raise RuntimeError(
            f"rtKernelLaunch failed for '{kernel_name}': "
            f"rtError={ret}, syncError={sync_ret}"
        )
