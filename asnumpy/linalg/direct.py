# *****************************************************************************
# Copyright (c) 2025 AISS and ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from ..lib.asnumpy_core import (
    dot as _ap_dot,
    inner as _ap_inner,
    outer as _ap_outer,
    vdot as _ap_vdot,
    matmul as _ap_matmul,
    einsum as _ap_einsum,
)
from ..utils import ndarray


def dot(a: ndarray, b: ndarray) -> ndarray:
    """
    Calculate the dot product of two arrays.

    This function computes the dot product based on the dimensionality of the input arrays `a` and `b`.
    - For 1-D arrays, it computes the inner product of vectors.
    - For 2-D arrays, it performs matrix multiplication.
    - For 0-D (scalar) inputs, it performs scalar multiplication.
    - For N-D arrays, it generally computes a sum product over the last axis of `a` and the second-to-last axis of `b`.

    Arguments
    ---------
    a : asnumpy.ndarray
        The first input array.
    b : asnumpy.ndarray
        The second input array.

    Returns
    -------
    asnumpy.ndarray
        The dot product of `a` and `b`.
        If both inputs are scalars or 1-D arrays, a scalar is returned.
        Otherwise, an array is returned. The result may be allocated on the accelerator device.

    See Also
    --------
    numpy.dot
    asnumpy.matmul
    asnumpy.tensordot

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.dot(5, 6)
    30
    >>> vec1 = ap.array([1, 2])
    >>> vec2 = ap.array([3, 4])
    >>> ap.dot(vec1, vec2)
    11
    >>> mat1 = ap.array([[1, 2], [3, 4]])
    >>> mat2 = ap.array([[5, 6], [7, 8]])
    >>> ap.dot(mat1, mat2)
    array([[19, 22],
           [43, 50]])
    """
    return ndarray(_ap_dot(a, b))


def inner(a: ndarray, b: ndarray) -> ndarray:
    """
    Compute the inner product of two arrays.

    Calculates the inner product by summing the product of elements 
    over the last dimensions of the input arrays `a` and `b`.
    For 1-D arrays, this is equivalent to the vector dot product.
    For higher dimensions, it computes the sum product over the last axis of both arrays.

    Arguments
    ---------
    a : asnumpy.ndarray
        The first input array.
    b : asnumpy.ndarray
        The second input array. The last dimension must match the last dimension of `a`.

    Returns
    -------
    asnumpy.ndarray
        The inner product of the arrays.
        The shape of the output is `a.shape[:-1] + b.shape[:-1]`.

    See Also
    --------
    numpy.inner
    asnumpy.dot
    asnumpy.tensordot

    Examples
    --------
    >>> import asnumpy as ap
    >>> vec1 = ap.array([2, 3])
    >>> vec2 = ap.array([4, 5])
    >>> ap.inner(vec1, vec2)
    23
    >>> a = ap.arange(6).reshape((2, 3))
    >>> b = ap.array([1, 2, 3])
    >>> ap.inner(a, b)
    array([14, 32])
    """
    return ndarray(_ap_inner(a, b))


def outer(a: ndarray, b: ndarray) -> ndarray:
    """
    Compute the outer product of two vectors.

    Calculates the outer product of two vectors `a` and `b`.
    If the inputs are not 1-D, they are flattened before computation.
    The result is a matrix where the element at `(i, j)` is the product of `a[i]` and `b[j]`.

    Arguments
    ---------
    a : asnumpy.ndarray
        The first input vector. Flattened if not 1-D.
    b : asnumpy.ndarray
        The second input vector. Flattened if not 1-D.

    Returns
    -------
    asnumpy.ndarray
        The outer product matrix.
        If `a` has size M and `b` has size N, the result has shape (M, N).

    See Also
    --------
    numpy.outer
    asnumpy.inner
    asnumpy.einsum

    Examples
    --------
    >>> import asnumpy as ap
    >>> vec1 = ap.array([1, 2, 3])
    >>> vec2 = ap.array([4, 5])
    >>> ap.outer(vec1, vec2)
    array([[ 4,  5],
           [ 8, 10],
           [12, 15]])
    """
    return ndarray(_ap_outer(a, b))


def vdot(a: ndarray, b: ndarray) -> ndarray:
    """
    Compute the dot product of two vectors, conjugating the first.

    This function computes the dot product of two vectors.
    Unlike `dot`, it flattens the input arrays into 1-D vectors first.
    If the first argument `a` is complex, its complex conjugate is used for the calculation.

    Arguments
    ---------
    a : asnumpy.ndarray
        The first input array. Flattened if not 1-D.
        If complex, the complex conjugate is used.
    b : asnumpy.ndarray
        The second input array. Flattened if not 1-D.

    Returns
    -------
    asnumpy.ndarray
        The dot product of the vectors.
        The result is a scalar.

    See Also
    --------
    numpy.vdot
    asnumpy.dot

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1+1j, 2+2j])
    >>> b = ap.array([1+2j, 3+4j])
    >>> ap.vdot(a, b)
    (17+3j)
    >>> ap.vdot(b, a)
    (17-3j)
    """
    return ndarray(_ap_vdot(a, b))


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Compute the matrix product of two arrays.

    This function implements the matrix product of two arrays.
    It supports standard matrix multiplication for 2-D arrays, 
    vector-matrix multiplication, and broadcasting for higher-dimensional arrays.
    Unlike `dot`, it does not support scalar multiplication.

    Arguments
    ---------
    x1 : asnumpy.ndarray
        The first input array. Scalars are not allowed.
    x2 : asnumpy.ndarray
        The second input array. Scalars are not allowed.

    Returns
    -------
    asnumpy.ndarray
        The matrix product of the inputs.
        The result is a scalar only if both inputs are 1-D vectors.

    See Also
    --------
    numpy.matmul
    asnumpy.dot
    asnumpy.tensordot
    asnumpy.einsum

    Examples
    --------
    >>> import asnumpy as ap
    >>> mat1 = ap.array([[1, 2], [3, 4]])
    >>> mat2 = ap.array([[5, 6], [7, 8]])
    >>> ap.matmul(mat1, mat2)
    array([[19, 22],
           [43, 50]])

    >>> vec = ap.array([1, 2])
    >>> ap.matmul(mat1, vec)
    array([ 5, 11])
    """
    return ndarray(_ap_matmul(x1, x2))


def einsum(subscripts: str, *operands: ndarray) -> ndarray:
    """
    Evaluate the Einstein summation convention on the operands.

    Performs multi-dimensional linear algebraic array operations using the Einstein summation convention.
    This allows for a concise representation of many common operations 
    like dot products, traces, and tensor contractions.

    Arguments
    ---------
    subscripts : str
        A string specifying the subscripts for summation.
        It consists of comma-separated subscript labels.
        If '->' is included, it explicitly defines the output subscripts.
        Otherwise, an implicit calculation is performed.
    *operands : asnumpy.ndarray
        The arrays to operate on.

    Returns
    -------
    asnumpy.ndarray
        The result of the Einstein summation.

    See Also
    --------
    numpy.einsum
    asnumpy.dot
    asnumpy.inner
    asnumpy.outer
    asnumpy.tensordot

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.arange(9).reshape(3, 3)
    >>> ap.einsum('ii', a)  # Trace
    12
    >>> ap.einsum('ii->i', a)  # Diagonal
    array([0, 4, 8])
    >>> b = ap.arange(3)
    >>> ap.einsum('ij,j', a, b)  # Matrix-vector multiplication
    array([ 5, 14, 23])
    """
    return ndarray(_ap_einsum(subscripts, *operands))


_direct_all_ = [
    "dot",
    "einsum",
    "inner",
    "matmul",
    "outer",
    "vdot",
]
