# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from typing import Optional, Union, Sequence
from .lib.asnumpy_core.sorting import sort as _ap_sort
from .utils import ndarray


def sort(a: ndarray, axis: Optional[int] = -1, stable: bool = False) -> ndarray:
    """
    Arrange array elements in ascending order.

    This function produces a new array with elements arranged from smallest to largest along the specified axis.
    If no axis is given, the last axis is used by default. Flattening occurs when axis=None.
    The `stable` flag ensures that the relative order of equal elements is preserved when set to True.

    Arguments
    ---------
    a : asnumpy.ndarray
        The input array whose elements will be rearranged.
    axis : int or None, optional
        Axis along which to sort. Default is -1 (last axis). If None, the array is flattened.
    stable : bool, optional
        Whether to perform a stable sort that preserves the order of equal elements.

    Returns
    -------
    asnumpy.ndarray
        A new array with elements sorted along the specified axis. Shape matches the input except when flattened.

    See Also
    --------
    numpy.sort

    Notes
    -----
    AsNumPy does not currently implement `kind` or `order` parameters from NumPy. 
    Use the `stable` boolean to control sorting stability.

    Examples
    --------
    >>> import asnumpy as ap
    >>> arr = ap.array([[3, 1], [2, 4]])
    >>> ap.sort(arr)
    array([[1, 3],
           [2, 4]])
    >>> ap.sort(arr, axis=0, stable=True)
    array([[2, 1],
           [3, 4]])
    """
    return ndarray(_ap_sort(a, axis, stable))
