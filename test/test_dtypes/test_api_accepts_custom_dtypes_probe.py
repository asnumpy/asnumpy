from __future__ import annotations

import os
import json
import subprocess
import sys
import textwrap

import pytest
from loguru import logger


_VERBOSE = os.getenv("ASNUMPY_TEST_VERBOSE", "0") == "1"


def _env_supports_asnumpy_api() -> bool:
    """
    High-level asnumpy APIs allocate device memory. Probe in a subprocess so we can
    skip cleanly on machines without a working Ascend runtime/device.
    """
    code = textwrap.dedent(
        """
        import numpy as np
        import asnumpy as ap

        x = ap.zeros((1,), dtype=ap.float32)
        print("ok", x.dtype, x.aclDtype)
        """
    ).strip()
    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return proc.returncode == 0 and (proc.stdout or "").startswith("ok")


_CUSTOM_DTYPES: list[tuple[str, int]] = [
    ("bfloat16", 27),          # ACL_BF16
    ("float8_e5m2", 35),       # ACL_FLOAT8_E5M2
    ("float8_e4m3fn", 36),     # ACL_FLOAT8_E4M3FN
    ("float8_e8m0", 37),       # ACL_FLOAT8_E8M0
    ("float6_e3m2fn", 38),     # ACL_FLOAT6_E3M2
    ("float6_e2m3fn", 39),     # ACL_FLOAT6_E2M3
    ("float4_e2m1fn", 40),     # ACL_FLOAT4_E2M1
    ("float4_e1m2fn", 41),     # ACL_FLOAT4_E1M2
    ("int4", 29),              # ACL_INT4
    ("uint1", 30),             # ACL_UINT1
]


@pytest.mark.parametrize(("name", "expected_acl_enum"), _CUSTOM_DTYPES)
def test_high_level_api_accepts_custom_dtype(name: str, expected_acl_enum: int) -> None:
    """
    Integration probe: ensure asnumpy "public API" can accept custom dtypes.

    We run in a subprocess because these APIs allocate device memory and may crash
    on environments without a usable Ascend runtime/device.
    """
    if not _env_supports_asnumpy_api():
        pytest.skip("当前环境不可用 Ascend runtime/device，跳过 asnumpy API 集成探测")

    code = textwrap.dedent(
        f"""
        import json
        import numpy as np
        import asnumpy as ap

        dt = ap.{name}
        expected_acl = {expected_acl_enum}

        def _ok(arr):
            if arr is None:
                return False
            if int(arr.aclDtype) != expected_acl:
                return False
            if arr.dtype != np.dtype(dt):
                return False
            return True

        supported = []
        results = {{}}

        # Strong requirement: empty must succeed for dtype acceptance + allocation.
        e = ap.empty((2, 3), dtype=dt)
        assert _ok(e)
        results["empty"] = "supported"
        supported.append("empty")

        # Weak / backend-dependent APIs (may return None due to @logger.catch).
        candidates = {{
            "zeros": lambda: ap.zeros((2, 3), dtype=dt),
            "ones": lambda: ap.ones((2, 3), dtype=dt),
            "full": lambda: ap.full((2, 3), value=1, dtype=dt),
            "eye": lambda: ap.eye(3, dtype=dt),
            "identity": lambda: ap.identity(3, dtype=dt),
            "linspace": lambda: ap.linspace(0.0, 1.0, steps=6, dtype=dt),
        }}

        for k, fn in candidates.items():
            try:
                arr = fn()
            except Exception as ex:
                results[k] = "error:" + type(ex).__name__
                continue
            if _ok(arr):
                supported.append(k)
                results[k] = "supported"
            else:
                # either None (caught) or wrong dtype/acl mapping
                results[k] = "unsupported"

        # *_like APIs: require a prototype array; we pass dtype override.
        proto = ap.empty((2, 3), dtype=ap.float32)
        like_candidates = {{
            "zeros_like": lambda: ap.zeros_like(proto, dtype=dt),
            "ones_like": lambda: ap.ones_like(proto, dtype=dt),
            "full_like": lambda: ap.full_like(proto, value=1, dtype=dt),
            "empty_like": lambda: ap.empty_like(proto, dtype=dt),
        }}
        for k, fn in like_candidates.items():
            try:
                arr = fn()
            except Exception as ex:
                results[k] = "error:" + type(ex).__name__
                continue
            if _ok(arr):
                supported.append(k)
                results[k] = "supported"
            else:
                results[k] = "unsupported"

        print(json.dumps({{"dtype": {name!r}, "expected_acl": expected_acl, "supported": supported, "results": results}}, ensure_ascii=False))
        """
    ).strip()

    proc = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if _VERBOSE:
        logger.info("subprocess stdout for {}:\n{}", name, (proc.stdout or "").strip())
        if (proc.stderr or "").strip():
            logger.warning("subprocess stderr for {}:\n{}", name, (proc.stderr or "").strip())

    assert proc.returncode == 0, f"subprocess failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    payload = json.loads((proc.stdout or "").strip())
    supported = payload.get("supported", [])
    results = payload.get("results", {})

    # Always log what is supported (user asked to display).
    logger.info("api support: {} -> {}", name, ", ".join(supported) if supported else "<none>")
    if _VERBOSE:
        logger.info("api details: {} -> {}", name, results)

