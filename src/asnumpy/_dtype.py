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
"""Data type routines, borrowed from NumPy.

asnumpy has no dtype system of its own: ``asnumpy.dtype`` *is* ``numpy.dtype`` and
``asnumpy.float32`` *is* ``numpy.float32``. Every promotion decision bottoms out in
``numpy.result_type`` / ``numpy.promote_types`` / ``numpy.can_cast``, so asnumpy cannot drift from
NumPy's rules. Only the device-specific parts are asnumpy's own: which dtypes ACL can represent
(see ``csrc/dtypes/dtype_table.cpp``) and which kernels accept them.

``result_type`` and ``can_cast`` are thin wrappers rather than re-exports for one reason: they must
unwrap an ``asnumpy.ndarray`` to its ``.dtype`` first. Handing the array itself to NumPy would
trigger a device-to-host transfer just to answer a question about metadata.
"""

__all__ = [
    "can_cast",
    "dtype",
    "finfo",
    "iinfo",
    "isdtype",
    "issubdtype",
    "promote_types",
    "result_type",
]

# Verbatim re-exports: these are pure metadata operations with no array involved, so NumPy's
# implementations apply unchanged.
from numpy import can_cast as _np_can_cast
from numpy import dtype as dtype
from numpy import finfo as finfo
from numpy import iinfo as iinfo
from numpy import isdtype as isdtype
from numpy import issubdtype as issubdtype
from numpy import promote_types as promote_types
from numpy import result_type as _np_result_type

from ._core import ndarray as _core_ndarray


def _unwrap(obj):
    """Reduce an asnumpy array to its dtype, leaving everything else untouched.

    Deliberately an isinstance check rather than ``getattr(obj, "dtype", ...)``: scalar *types*
    such as ``np.int32`` are classes whose ``dtype`` attribute is an unbound descriptor, not a
    dtype, so duck-typing would corrupt them. NumPy handles every other input itself.
    """
    return obj.dtype if isinstance(obj, _core_ndarray) else obj


def result_type(*arrays_and_dtypes):
    """Return the type that results from applying NumPy's promotion rules to the arguments.

    Mirrors :func:`numpy.result_type`, with asnumpy arrays reduced to their dtype first so that
    no device transfer occurs.
    """
    return _np_result_type(*[_unwrap(a) for a in arrays_and_dtypes])


def can_cast(from_, to, casting="safe"):
    """Return True if a cast between data types is possible under the given rule.

    Mirrors :func:`numpy.can_cast`, with asnumpy arrays reduced to their dtype first so that no
    device transfer occurs.
    """
    return _np_can_cast(_unwrap(from_), to, casting=casting)
