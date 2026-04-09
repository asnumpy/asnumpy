"""
Probe the new ACL-path API: asnumpy_core.array.zeros_like_acl(other, acl_type).

This is a script-style test (no pytest dependency) that runs each case in a
subprocess so that backend crashes (SIGSEGV) don't take down the test runner.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from loguru import logger


def _to_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    return str(x)


def _run_case(acl_type: int, label: str, timeout_s: int = 15) -> tuple[int, str, str]:
    code = textwrap.dedent(
        f"""
        import asnumpy as ap
        from asnumpy.lib import asnumpy_core

        proto = ap.zeros((4,), dtype=ap.float32)
        out = asnumpy_core.array.zeros_like_acl(proto, {acl_type})
        print({label!r}, "ok", "acl_dtype=", out.aclDtype, "dtype=", out.dtype)
        """
    ).strip()

    try:
        proc = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        stdout = _to_text(e.stdout)
        stderr = _to_text(e.stderr)
        return 124, stdout, stderr + f"\n[timeout] exceeded {timeout_s}s\n"


def run_all() -> None:
    timeout_s = int(os.getenv("ASNUMPY_PROBE_TIMEOUT_S", "15"))
    cases: list[tuple[str, int]] = [
        ("ACL_DT_UNDEFINED", -1),
        ("ACL_FLOAT", 0),
        ("ACL_FLOAT16", 1),
        ("ACL_INT8", 2),
        ("ACL_INT32", 3),
        ("ACL_UINT8", 4),
        ("ACL_INT16", 6),
        ("ACL_UINT16", 7),
        ("ACL_UINT32", 8),
        ("ACL_INT64", 9),
        ("ACL_UINT64", 10),
        ("ACL_DOUBLE", 11),
        ("ACL_BOOL", 12),
        ("ACL_STRING", 13),
        ("ACL_COMPLEX64", 16),
        ("ACL_COMPLEX128", 17),
        ("ACL_BF16", 27),
        ("ACL_INT4", 29),
        ("ACL_UINT1", 30),
        ("ACL_COMPLEX32", 33),
        ("ACL_HIFLOAT8", 34),      # 当前不支持该类型
        ("ACL_FLOAT8_E5M2", 35),   # 当前不支持该类型
        ("ACL_FLOAT8_E4M3FN", 36), # 当前不支持该类型
        ("ACL_FLOAT8_E8M0", 37),   # 当前不支持该类型
        ("ACL_FLOAT6_E3M2", 38),   # 当前不支持该类型
        ("ACL_FLOAT6_E2M3", 39),   # 当前不支持该类型
        ("ACL_FLOAT4_E2M1", 40),   # 当前不支持该类型
        ("ACL_FLOAT4_E1M2", 41),   # 当前不支持该类型
    ]

    logger.info("Probing zeros_like_acl via subprocess.")
    logger.info("=" * 60)

    for label, acl_type in cases:
        rc, out, err = _run_case(acl_type=acl_type, label=label, timeout_s=timeout_s)
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        logger.info("[{}] {} acl_type={}", status, label, acl_type)

        if out.strip():
            logger.info("stdout: {}", out.strip())
        if err.strip():
            tail = "\n".join(err.strip().splitlines()[-20:])
            logger.warning("stderr tail:\n{}", tail)

        logger.info("-" * 60)

    logger.info("Done.")


if __name__ == "__main__":
    run_all()

