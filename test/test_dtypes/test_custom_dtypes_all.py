"""
Systematic pytest coverage for all AsNumpy custom dtypes exported via:

- asnumpy.dtypes.<name>
- asnumpy.<name> (top-level alias)

We keep assertions focused on API/contract (export, np.dtype, scalar, getACLenum,
astype) rather than numerical exactness (which is dtype-dependent).
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
from loguru import logger

import asnumpy as ap


# Keep this list in sync with:
# - `asnumpy/__init__.py` -> `_ASNUMPY_CUSTOM_DTYPE_NAMES`
# - `python/bind_dtypes.cpp` -> ASNUMPY_DTYPE_BIND_LIST
# - `src/dtypes/reg.cpp` -> InitAndRegisterDtypes()
_CUSTOM_DTYPES: list[tuple[str, str]] = [
    # name, expected numpy dtype.kind
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

_VERBOSE = os.getenv("ASNUMPY_TEST_VERBOSE", "0") == "1"


@pytest.mark.parametrize(("name", "expected_kind"), _CUSTOM_DTYPES)
def test_dtype_is_exported_as_submodule_and_top_level_alias(name: str, expected_kind: str) -> None:
    dtypes_mod = importlib.import_module("asnumpy.dtypes")

    assert hasattr(dtypes_mod, name), f"asnumpy.dtypes.{name} 未导出"
    assert hasattr(ap, name), f"asnumpy.{name} 未直出（顶层别名缺失）"

    dtype_obj_from_submodule = getattr(dtypes_mod, name)
    dtype_obj_from_top_level = getattr(ap, name)
    assert dtype_obj_from_top_level is dtype_obj_from_submodule, f"ap.{name} 与 asnumpy.dtypes.{name} 不一致"

    # Must be recognized by NumPy dtype system.
    dt = np.dtype(dtype_obj_from_top_level)
    assert dt is not None
    assert dt.kind == expected_kind

    if _VERBOSE:
        logger.info("export ok: {} (kind='{}', itemsize={})", name, dt.kind, dt.itemsize)


@pytest.mark.parametrize(("name", "expected_kind"), _CUSTOM_DTYPES)
def test_dtype_scalar_and_getACLenum(name: str, expected_kind: str) -> None:
    dtype_obj = getattr(ap, name)

    # Scalar construction should work and provide getACLenum().
    scalar = dtype_obj(3.14 if expected_kind == "f" else 1)
    assert hasattr(scalar, "getACLenum"), f"{name} 标量缺少 getACLenum()"
    acl_enum = scalar.getACLenum()
    assert isinstance(acl_enum, int)

    # Qualified name should match the canonical module path.
    scalar_type = type(scalar)
    full_name = f"{scalar_type.__module__}.{scalar_type.__name__}"
    assert full_name == f"asnumpy.dtypes.{name}"

    if _VERBOSE:
        logger.info("scalar ok: {} (acl_enum={}, type={})", name, acl_enum, full_name)


@pytest.mark.parametrize(("name", "expected_kind"), _CUSTOM_DTYPES)
def test_dtype_array_astype_roundtrip(name: str, expected_kind: str) -> None:
    dtype_obj = getattr(ap, name)

    if expected_kind == "f":
        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        casted = src.astype(dtype_obj)
        assert casted.dtype == np.dtype(dtype_obj)

        recovered = casted.astype(np.float32)
        # Not asserting strict numerical fidelity, only that the conversion path works
        # and produces finite values in a reasonable range.
        assert recovered.shape == src.shape
        assert np.all(np.isfinite(recovered))
        assert np.max(np.abs(recovered)) < 1e6

        # bfloat16 is 2 bytes; other custom floats are expected to be 1 byte.
        if name == "bfloat16":
            assert casted.itemsize == 2
        else:
            assert casted.itemsize == 1

        if _VERBOSE:
            logger.info(
                "astype ok: {} (src=float32 -> custom -> float32, itemsize={})",
                name,
                casted.itemsize,
            )
    else:
        src = np.array([0, 1, 2, 3], dtype=np.int32)
        casted = src.astype(dtype_obj)
        assert casted.dtype == np.dtype(dtype_obj)

        recovered = casted.astype(np.int32)
        assert recovered.shape == src.shape

        if _VERBOSE:
            logger.info("astype ok: {} (src=int32 -> custom -> int32)", name)

