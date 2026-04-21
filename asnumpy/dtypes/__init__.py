# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
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

