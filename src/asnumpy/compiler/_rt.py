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

"""Minimal ctypes wrapper for CANN 9.x kernel launch.

Uses rtDevBinaryRegister + rtFunctionRegister + rtKernelLaunch (RTS path).
The .aicore_binary section ELF is extracted from the bisheng-compiled .o
and registered with RT_DEV_BINARY_MAGIC_ELF_AIVEC (0x41415246).
"""

import ctypes
import os

_ASCEND_HOME = os.environ.get(
    "ASCEND_TOOLKIT_HOME",
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
)

RT_DEV_BINARY_MAGIC_ELF_AIVEC = 0x41415246
FUNC_MODE_NORMAL = 0

_librt: ctypes.CDLL | None = None
_registry: dict[int, object] = {}         # bin_handle -> buffer
_name_refs: dict[str, bytes] = {}         # kernel_name -> bytes (ptr stability)


class _RtDevBinary(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("data", ctypes.c_void_p),
        ("length", ctypes.c_uint64),
    ]


def _get_rt() -> ctypes.CDLL:
    global _librt
    if _librt is None:
        _librt = ctypes.CDLL(f"{_ASCEND_HOME}/lib64/libruntime.so")
        _librt.rtDevBinaryRegister.argtypes = [
            ctypes.POINTER(_RtDevBinary), ctypes.POINTER(ctypes.c_void_p),
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
        # ACL for sync
        _libacl = ctypes.CDLL(f"{_ASCEND_HOME}/lib64/libascendcl.so")
        _libacl.aclrtSynchronizeDevice.argtypes = []
        _libacl.aclrtSynchronizeDevice.restype = ctypes.c_int
        _librt._libacl = _libacl
    return _librt


def register_binary(data: bytes) -> int:
    librt = _get_rt()
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    dev_bin = _RtDevBinary(
        magic=RT_DEV_BINARY_MAGIC_ELF_AIVEC, version=0,
        data=ctypes.cast(buf, ctypes.c_void_p), length=len(data),
    )
    handle = ctypes.c_void_p()
    ret = librt.rtDevBinaryRegister(ctypes.byref(dev_bin), ctypes.byref(handle))
    if ret != 0:
        raise RuntimeError(f"rtDevBinaryRegister failed: rtError={ret}")
    h = handle.value
    _registry[h] = buf
    return h


def register_function(bin_handle: int, kernel_name: str) -> None:
    librt = _get_rt()
    name = kernel_name.encode()
    _name_refs[kernel_name] = name
    ret = librt.rtFunctionRegister(
        ctypes.c_void_p(bin_handle), name, name, name, FUNC_MODE_NORMAL,
    )
    if ret != 0:
        raise RuntimeError(f"rtFunctionRegister failed for '{kernel_name}': rtError={ret}")


def launch_kernel(kernel_name: str, block_dim: int,
                  packed_args: list[bytes], stream: int = 0) -> None:
    librt = _get_rt()

    # 8-byte alignment for each arg
    aligned = []
    for a in packed_args:
        aligned.append(a)
        if len(a) % 8 != 0:
            aligned.append(b"\x00" * (8 - len(a) % 8))
    flat = b"".join(aligned)
    total = len(flat)
    args_buf = (ctypes.c_ubyte * total).from_buffer_copy(flat)

    name = _name_refs.get(kernel_name)
    if name is None:
        name = kernel_name.encode()
        _name_refs[kernel_name] = name

    ret = librt.rtKernelLaunch(name, block_dim, args_buf, total, None,
                               ctypes.c_void_p(stream))
    if ret != 0:
        raise RuntimeError(f"rtKernelLaunch failed for '{kernel_name}': rtError={ret}")

    sync_ret = librt._libacl.aclrtSynchronizeDevice()
    if sync_ret != 0:
        raise RuntimeError(f"aclrtSynchronizeDevice failed: aclError={sync_ret}")


def unregister_binary(bin_handle: int) -> None:
    librt = _get_rt()
    ret = librt.rtDevBinaryUnRegister(ctypes.c_void_p(bin_handle))
    if ret != 0:
        import sys
        print(f"[asnumpy] rtDevBinaryUnRegister warning: rtError={ret}", file=sys.stderr)
    _registry.pop(bin_handle, None)
