from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
from loguru import logger

import asnumpy as ap
import numpy as np


_VERBOSE = os.getenv("ASNUMPY_TEST_VERBOSE", "0") == "1"


def _env_supports_npuarray_ctor() -> bool:
    """
    Constructing NPUArray allocates device memory. Probe in a subprocess to avoid
    crashing the test runner in environments without a working Ascend runtime/device.
    """
    code = textwrap.dedent(
        """
        import numpy as np
        from asnumpy.lib import asnumpy_core

        _ = asnumpy_core.ndarray([1], np.dtype(np.float32))
        print("ok")
        """
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc.returncode == 0 and "ok" in (proc.stdout or "")


_CUSTOM_DTYPES: list[tuple[str, str]] = [
    ("float8_e5m2", "f"),
    ("float8_e4m3fn", "f"),
    ("float8_e8m0", "f"),
    ("bfloat16", "f"),
    ("float6_e2m3fn", "f"),
    ("float6_e3m2fn", "f"),
    ("float4_e2m1fn", "f"),
    ("float4_e1m2fn", "f"),
    ("int4", "i"),
    ("uint1", "u"),
]


@pytest.mark.parametrize(("name", "kind"), _CUSTOM_DTYPES)
def test_scalar_getaclenum_matches_npuarray_acl_dtype(name: str, kind: str) -> None:
    """
    Consistency check / debug aid:

    - Scalar path: scalar.getACLenum() (registered on scalar type)
    - DType->ACL path: NPUArray(shape, py::dtype) -> GetACLDataType(dtype) -> aclDtype

    They should agree on the same ACL enumeration value.
    """
    if not _env_supports_npuarray_ctor():
        pytest.skip("当前环境不可用 Ascend runtime/device，跳过一致性集成校验")

    dtype_obj = getattr(ap, name)
    scalar = dtype_obj(1.0 if kind == "f" else 1)
    scalar_acl = int(scalar.getACLenum())

    code = textwrap.dedent(
        f"""
        import numpy as np
        import asnumpy as ap
        from asnumpy.lib import asnumpy_core

        arr = asnumpy_core.ndarray([2, 3], np.dtype(ap.{name}))
        print(int(arr.aclDtype))
        """
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"subprocess failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    array_acl = int(proc.stdout.strip())
    if _VERBOSE:
        logger.info("acl enum match: {} scalar={} array={}", name, scalar_acl, array_acl)
    assert array_acl == scalar_acl

