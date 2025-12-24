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
import numpy as np
from ..lib.asnumpy_core.linalg import (
    det as ap_det,
    inv as ap_inv,
    matrix_power as ap_matrix_power,
    norm as ap_norm,
    slogdet as ap_slogdet,
)
from ..utils import ndarray


def matrix_power(a: ndarray, n: int) -> ndarray:
    """
    Raise a square matrix to the (integer) power `n`.

    For positive integers `n`, the power is computed by repeated matrix
    squarings and matrix multiplications. If ``n == 0``, the identity matrix
    of the same shape as ``a`` is returned. If ``n < 0``, the inverse
    is computed and then raised to the ``abs(n)``.

    Parameters
    ----------
    a : ndarray
        Matrix to be "powered".
    n : int
        The exponent can be any integer or long integer, positive,
        negative, or zero.

    Returns
    -------
    a**n : ndarray
        The return value is the same shape and type as ``a``;
        if the exponent is positive or zero then the type of the
        elements is the same as those of ``a``. If the exponent is
        negative the elements are floating-point.

    See Also
    --------
    numpy.linalg.matrix_power
    """
    return ndarray(ap_matrix_power(a, n))


def qr(a: ndarray, mode: str = "reduced") -> Union[ndarray, tuple]:
    """
    Compute the qr factorization of a matrix.

    Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
    upper-triangular.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.
    mode : {'reduced', 'complete', 'r', 'raw'}, optional
        If K = min(M, N), then

        * 'reduced'  : returns q, r with dimensions (M, K), (K, N) (default)
        * 'complete' : returns q, r with dimensions (M, M), (M, N)
        * 'r'        : returns r only with dimensions (K, N)
        * 'raw'      : returns h, tau with dimensions (N, M), (K,)

        Note that array h returned in 'raw' mode is transposed for calling
        Fortran. The 'economic' mode is deprecated.  The modes 'full' and
        'economic' may be passed using the first letter for backwards
        compatibility, but all others must be spelled out. See the Notes
        for more explanation.

    Returns
    -------
    q : ndarray of float or complex, optional
        A matrix with orthonormal columns. When mode = 'complete' the
        result is an orthogonal/unitary matrix depending on whether or not
        a is real/complex. The determinant may be either +/- 1 in that
        case.
    r : ndarray of float or complex, optional
        The upper-triangular matrix.

    See Also
    --------
    numpy.linalg.qr
    """
    return ndarray.from_numpy(np.linalg.qr(a, mode))


def norm(
    a: ndarray,
    ord: Optional[Union[str, int, float]] = None,
    axis: Optional[Union[int, Sequence[int]]] = None,
    keepdims: bool = False,
) -> ndarray:
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Input array.  If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of
        ``a.ravel`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned. The default
        is None.

        .. versionadded:: 1.8.0

    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `a`.

        .. versionadded:: 1.10.0

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    See Also
    --------
    numpy.linalg.norm
    """
    return ndarray(ap_norm(a, ord, axis, keepdims))


def det(a: ndarray) -> ndarray:
    """
    Compute the determinant of an array.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input array to compute determinants for.

    Returns
    -------
    det : (...) ndarray
        Determinant of `a`.

    See Also
    --------
    slogdet : Another way to represent the determinant, more suitable
      for large matrices where underflow/overflow may occur.
    numpy.linalg.det
    """
    return ndarray(ap_det(a))


def slogdet(a: ndarray) -> tuple:
    """
    Compute the sign and (natural) logarithm of the determinant of an array.

    If an array has a very small or very large determinant, then a call to
    `det` may overflow or underflow. This routine is more robust against such
    issues, because it computes the logarithm of the determinant rather than
    the determinant itself.

    Parameters
    ----------
    a : (..., M, M) array_like
        Input array, has to be a square 2-D array.

    Returns
    -------
    sign : (...) ndarray
        A number representing the sign of the determinant. For a real matrix,
        this is 1, 0, or -1. For a complex matrix, this is a complex number
        with absolute value 1 (i.e., it is on the unit circle), or else 0.
    logdet : (...) ndarray
        The natural log of the absolute value of the determinant.

    If the determinant is zero, then `sign` will be 0 and `logdet` will be
    -Inf. In all cases, the determinant is equal to ``sign * exp(logdet)``.

    See Also
    --------
    det
    numpy.linalg.slogdet
    """
    return ap_slogdet(a)


def inv(a: ndarray) -> ndarray:
    """
    Compute the (multiplicative) inverse of a matrix.

    Given a square matrix `a`, return the matrix `ainv` satisfying
    ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``.

    Parameters
    ----------
    a : (..., M, M) array_like
        Matrix to be inverted.

    Returns
    -------
    ainv : (..., M, M) ndarray
        (Multiplicative) inverse of the matrix `a`.

    Raises
    ------
    LinAlgError
        If `a` is not square or inversion fails.

    See Also
    --------
    numpy.linalg.inv
    """
    return ndarray(ap_inv(a))
