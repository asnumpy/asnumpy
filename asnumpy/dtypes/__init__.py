from __future__ import annotations

import sys
from typing import Any

from ..lib.asnumpy_core import dtypes as _core_dtypes


def __getattr__(name: str) -> Any:
    return getattr(_core_dtypes, name)


def __dir__() -> list[str]:
    public = [n for n in dir(_core_dtypes) if not n.startswith("_")]
    return sorted(set(public + list(globals().keys())))


# Ensure the canonical import path exists in sys.modules.
sys.modules.setdefault("asnumpy.dtypes", sys.modules[__name__])

