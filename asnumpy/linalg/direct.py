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
    Dot product of two arrays.

    * If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).
    * If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.
    * If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``numpy.multiply(a, b)`` or ``a * b`` is
      preferred.
    * If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.
    * If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of
      `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Parameters
    ----------
    a : ndarray
        First argument.
    b : ndarray
        Second argument.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.
    matmul : '@' operator as method with out parameter.
    numpy.dot

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.dot(3, 4)
    12
    >>> ap.dot([2j, 3j], [2j, 3j])
    (-13+0j)
    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> ap.dot(a, b)
    array([[4, 1],
           [2, 2]])
    """
    return ndarray(_ap_dot(a, b))


def inner(a: ndarray, b: ndarray) -> ndarray:
    """
    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : ndarray
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        `out.shape = a.shape[:-1] + b.shape[:-1]`

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : General function for matrix multiplication.
    einsum : Einstein summation convention.
    numpy.inner

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1,2,3])
    >>> b = ap.array([0,1,0])
    >>> ap.inner(a, b)
    2
    >>> a = ap.arange(24).reshape((2,3,4))
    >>> b = ap.arange(4)
    >>> ap.inner(a, b)
    array([[ 14,  38,  62],
           [ 86, 110, 134]])
    """
    return ndarray(_ap_inner(a, b))


def outer(a: ndarray, b: ndarray) -> ndarray:
    """
    Compute the outer product of two vectors.

    Given two vectors, ``a = [a0, a1, ..., aM]`` and
    ``b = [b0, b1, ..., bN]``,
    the outer product [1]_ is::

      [[a0*b0  a0*b1 ... a0*bN ]
       [a1*b0    .
       [ ...          .
       [aM*b0            aM*bN ]]

    Parameters
    ----------
    a : (M,) ndarray
        First input vector.  Input is flattened if
        not already 1-dimensional.
    b : (N,) ndarray
        Second input vector.  Input is flattened if
        not already 1-dimensional.

    Returns
    -------
    out : (M, N) ndarray
        ``out[i, j] = a[i] * b[j]``

    See Also
    --------
    inner
    einsum : Einstein summation convention.
    numpy.outer

    References
    ----------
    .. [1] : G. H. Golub and C. F. Van Loan, *Matrix Computations*, 3rd
             ed., Baltimore, MD, Johns Hopkins University Press, 1996,
             pg. 8.

    Examples
    --------
    >>> import asnumpy as ap
    >>> rl = ap.outer(ap.ones((5,)), ap.linspace(-2, 2, 5))
    >>> rl
    array([[-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.],
           [-2., -1.,  0.,  1.,  2.]])
    """
    return ndarray(_ap_outer(a, b))


def vdot(a: ndarray, b: ndarray) -> ndarray:
    """
    Return the dot product of two vectors.

    The vdot(a, b) function handles complex numbers differently than dot(a, b).
    If the first argument is complex the complex conjugate of the first argument
    is used for the calculation of the dot product.

    Note that `vdot` handles multidimensional arrays differently than `dot`:
    it does *not* perform a matrix product, but flattens input arguments
    to 1-D vectors first. Consequently, it should only be used for vectors.

    Parameters
    ----------
    a : ndarray
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : ndarray
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.  Can be an int, float, or complex
        depending on the types of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
          first argument.
    numpy.vdot

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.array([1+2j,3+4j])
    >>> b = ap.array([5+6j,7+8j])
    >>> ap.vdot(a, b)
    (70-8j)
    >>> ap.vdot(b, a)
    (70+8j)
    """
    return ndarray(_ap_vdot(a, b))


def matmul(x1: ndarray, x2: ndarray) -> ndarray:
    """
    Matrix product of two arrays.

    Parameters
    ----------
    x1, x2 : ndarray
        Input arrays, scalars not allowed.

    Returns
    -------
    out : ndarray
        The matrix product of the inputs.
        This is a scalar only when both x1, x2 are 1-d vectors.

    Raises
    ------
    ValueError
        If the last dimension of `x1` is not the same size as
        the second-to-last dimension of `x2`.

        If a scalar value is passed in.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.
    dot : Alternative matrix product with different broadcasting rules.
    numpy.matmul

    Examples
    --------
    For 2-D arrays it is the matrix product:

    >>> import asnumpy as ap
    >>> a = ap.array([[1, 0],
    ...               [0, 1]])
    >>> b = ap.array([[4, 1],
    ...               [2, 2]])
    >>> ap.matmul(a, b)
    array([[4, 1],
           [2, 2]])

    For 2-D mixed with 1-D, the result is the usual.

    >>> a = ap.array([[1, 0],
    ...               [0, 1]])
    >>> b = ap.array([1, 2])
    >>> ap.matmul(a, b)
    array([1, 2])
    >>> ap.matmul(b, a)
    array([1, 2])
    """
    return ndarray(_ap_matmul(x1, x2))


def einsum(subscripts: str, *operands: ndarray) -> ndarray:
    """
    Evaluates the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional,
    linear algebraic array operations can be represented in a simple fashion.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of ndarray
        These are the arrays for the operation.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.

    See Also
    --------
    dot, inner, outer, tensordot, linalg.einsum_path
    numpy.einsum

    Examples
    --------
    >>> import asnumpy as ap
    >>> a = ap.arange(25).reshape(5,5)
    >>> b = ap.arange(5)
    >>> c = ap.arange(6).reshape(2,3)

    Trace of a matrix:

    >>> ap.einsum('ii', a)
    60
    >>> ap.trace(a)
    60

    Extract the diagonal (requires explicit form):

    >>> ap.einsum('ii->i', a)
    array([ 0,  6, 12, 18, 24])
    >>> ap.diag(a)
    array([ 0,  6, 12, 18, 24])

    Sum over an axis (requires explicit form):

    >>> ap.einsum('ij->i', a)
    array([ 10,  35,  60,  85, 110])
    >>> ap.sum(a, axis=1)
    array([ 10,  35,  60,  85, 110])
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
