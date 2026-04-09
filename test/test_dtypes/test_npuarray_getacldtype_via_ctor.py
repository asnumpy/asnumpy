from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import asnumpy as ap
import numpy as np


def _env_supports_npuarray_ctor() -> bool:
    """
    Best-effort probe: constructing NPUArray allocates device memory.
    In environments without a working Ascend runtime/device, this may fail or crash.
    We probe in a subprocess to avoid taking down the test runner.
    """
    code = textwrap.dedent(
        """
        import numpy as np
        from asnumpy.lib import asnumpy_core

        # Use a standard dtype to minimize dependency on custom dtype registry.
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


@pytest.mark.parametrize(
    ("dtype_obj", "expected_acl_enum"),
    [
        (ap.bfloat16, 27),          # ACL_BF16
        (ap.float8_e5m2, 35),       # ACL_FLOAT8_E5M2
        (ap.float8_e4m3fn, 36),     # ACL_FLOAT8_E4M3FN
        (ap.float8_e8m0, 37),       # ACL_FLOAT8_E8M0
        (ap.float6_e3m2fn, 38),     # ACL_FLOAT6_E3M2
        (ap.float6_e2m3fn, 39),     # ACL_FLOAT6_E2M3
        (ap.float4_e2m1fn, 40),     # ACL_FLOAT4_E2M1
        (ap.float4_e1m2fn, 41),     # ACL_FLOAT4_E1M2
        (ap.int4, 29),              # ACL_INT4
        (ap.uint1, 30),             # ACL_UINT1
    ],
)
def test_npuarray_ctor_accepts_custom_dtype_and_maps_to_acl_enum(dtype_obj, expected_acl_enum: int) -> None:
    """
    API path coverage:
      Python -> asnumpy_core.ndarray(shape, dtype) -> NPUArray(shape, py::dtype)
            -> NPUArray::GetACLDataType(py::dtype) -> aclDataType enum
    """
    from asnumpy.lib import asnumpy_core

    if not _env_supports_npuarray_ctor():
        pytest.skip("当前环境不可用 Ascend runtime/device，跳过 NPUArray 构造集成测试")

    # Run in subprocess to avoid crashing the main test runner on backend issues.
    code = textwrap.dedent(
        f"""
        import numpy as np
        import asnumpy as ap
        from asnumpy.lib import asnumpy_core

        arr = asnumpy_core.ndarray([2, 3], np.dtype(ap.{dtype_obj.__name__}))
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
    got = int(proc.stdout.strip())
    assert got == expected_acl_enum

