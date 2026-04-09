from __future__ import annotations

import pytest
from loguru import logger


def _env_supports_asnumpy_api() -> bool:
    try:
        import asnumpy as ap

        x = ap.zeros((1,), dtype=ap.float32)
        return x is not None
    except Exception:
        return False


def test_api_support_report_bfloat16_only() -> None:
    """
    Report-style test, but only for currently supported dtype: bfloat16.
    No unsupported APIs are probed here.
    """
    if not _env_supports_asnumpy_api():
        pytest.skip("当前环境不可用 Ascend runtime/device，跳过 asnumpy API(bfloat16) 测试")

    import numpy as np
    import asnumpy as ap

    dt = ap.bfloat16
    expected_acl = 27

    def _ok(arr) -> None:
        assert arr is not None
        assert int(arr.aclDtype) == expected_acl
        assert arr.dtype == np.dtype(dt)

    e = ap.empty((2, 3), dtype=dt)
    _ok(e)
    z = ap.zeros((2, 3), dtype=dt)
    _ok(z)
    o = ap.ones((2, 3), dtype=dt)
    _ok(o)

    proto = ap.empty((2, 3), dtype=ap.float32)
    el = ap.empty_like(proto, dtype=dt)
    _ok(el)
    zl = ap.zeros_like(proto, dtype=dt)
    _ok(zl)
    ol = ap.ones_like(proto, dtype=dt)
    _ok(ol)

    logger.info("✓ bfloat16 支持: empty, zeros, ones, empty_like, zeros_like, ones_like")

