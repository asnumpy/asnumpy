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

from typing import Optional, Union
import numpy as np
from ..lib.asnumpy_core.linalg import (
    det as _det,
    inv as _inv,
    matrix_power as _matrix_power,
    norm as _norm,
    slogdet as _slogdet,
)
from ..utils import ndarray
from .._types import ArrayLike, AxisLike


def matrix_power(a: ArrayLike, n: int) -> ndarray:
    """
    Compute the integer power of a square matrix.

    This operation applies repeated matrix multiplication to obtain
    an integer power of the input matrix. Special cases such as zero
    and negative exponents are handled according to linear algebra
    conventions.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input square matrix.
    n : int
        Integer exponent applied to the matrix.

    Returns
    -------
    asnumpy.ndarray
        Resulting matrix after applying the specified power operation.

    See Also
    --------
    numpy.linalg.matrix_power

    Notes
    -----
    - A zero exponent produces an identity matrix with the same shape as ``a``.
    - Negative exponents involve computing the matrix inverse, which may
      promote the result to a floating-point data type.
    - Computation may be executed on an accelerator device depending on
      the asnumpy runtime configuration.

    Examples
    --------
    >>> import asnumpy as ap
    >>> m = ap.array([[1, 2], [0, 1]])
    >>> ap.linalg.matrix_power(m, 2)
    array([[1, 4],
           [0, 1]])
    """
    return ndarray(_matrix_power(a, n))


def qr(a: ArrayLike, mode: str = "reduced") -> Union[ndarray, tuple]:
    """
    Perform QR factorization of a matrix.

    Factor the input matrix into an orthonormal matrix `q` and an
    upper-triangular matrix `r`. Different factorization modes
    control which outputs are returned.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input matrix to be factored.
    mode : str, optional
        Factorization mode. Options include:
        - 'reduced' (default): returns `q` and `r` with minimal size
        - 'complete': returns full-size `q` and `r`
        - 'r': returns `r` only
        - 'raw': returns raw matrices `h` and `tau` used in computation

    Returns
    -------
    asnumpy.ndarray or tuple
        Depending on `mode`, returns `r` alone, a tuple `(q, r)`, or
        raw matrices `(h, tau)`.

    See Also
    --------
    numpy.linalg.qr

    Notes
    -----
    - Raw mode returns a transposed matrix for Fortran compatibility.
    - Computation may be performed on an accelerator device depending
      on asnumpy runtime.

    Examples
    --------
    >>> import asnumpy as ap
    >>> mtx = ap.array([[1., 2.], [3., 4.]])
    >>> q, r = ap.linalg.qr(mtx)
    >>> q.shape, r.shape
    ((2, 2), (2, 2))
    """
    return ndarray.from_numpy(np.linalg.qr(a, mode))


def norm(
    a: ArrayLike,
    ord: Optional[Union[str, int, float]] = None,
    axis: AxisLike = None,
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the norm of a vector or matrix.

    Calculates vector or matrix norms with various orders, including
    Frobenius and nuclear norms. The type of norm is determined by
    the `ord` and `axis` parameters.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input vector or matrix.
    ord : int, float, or str, optional
        Order of the norm (e.g., 2, inf, -inf, 'fro', 'nuc'). Default is None.
    axis : int or tuple of ints, optional
        Axis or axes along which to compute the norm.
    keepdims : bool, optional
        If True, reduced axes are kept as size-one dimensions.

    Returns
    -------
    asnumpy.ndarray
        Norm values. Scalar if axis=None, otherwise array with reduced dimensions.

    See Also
    --------
    numpy.linalg.norm

    Notes
    -----
    - Vector norms and matrix norms behave differently depending on `axis`.
    - Computation may use accelerator devices if configured.

    Examples
    --------
    >>> import asnumpy as ap
    >>> v = ap.array([3., 4.])
    >>> ap.linalg.norm(v)
    5.0
    >>> M = ap.array([[1., 2.], [3., 4.]])
    >>> ap.linalg.norm(M, 'fro')
    5.477225575051661
    """
    return ndarray(_norm(a, ord, axis, keepdims))


def det(a: ArrayLike) -> ndarray:
    """
    Compute the determinant of a square matrix.

    Returns a scalar or array containing determinants of input matrices.

    Arguments
    ---------
    a : asnumpy.ndarray
        Square matrix or batch of square matrices.

    Returns
    -------
    asnumpy.ndarray
        Determinant value(s) for each matrix.

    See Also
    --------
    numpy.linalg.det
    slogdet

    Notes
    -----
    - Use `slogdet` for better numerical stability on very large or small determinants.
    - Computation may run on an accelerator device.

    Examples
    --------
    >>> import asnumpy as ap
    >>> mtx = ap.array([[1., 2.], [3., 4.]])
    >>> ap.linalg.det(mtx)
    -2.0
    """
    return ndarray(_det(a))


def slogdet(a: ArrayLike) -> tuple:
    """
    Compute the sign and logarithm of the determinant.

    Provides a numerically stable way to evaluate determinants, especially
    for very large or small values.

    Arguments
    ---------
    a : asnumpy.ndarray
        Input square matrix or batch of square matrices.

    Returns
    -------
    sign : asnumpy.ndarray
        Sign of the determinant.
    logdet : asnumpy.ndarray
        Natural logarithm of the absolute value of the determinant.

    See Also
    --------
    det
    numpy.linalg.slogdet

    Notes
    -----
    - If a determinant is zero, `sign` is 0 and `logdet` is -Inf.
    - The original determinant can be recovered as `sign * exp(logdet)`.

    Examples
    --------
    >>> import asnumpy as ap
    >>> M = ap.array([[1., 2.], [3., 4.]])
    >>> s, ld = ap.linalg.slogdet(M)
    >>> s, ld
    (-1.0, 0.6931471805599453)
    """
    return _slogdet(a)


def inv(a: ArrayLike) -> ndarray:
    """
    Compute the multiplicative inverse of a square matrix.

    Returns a matrix that satisfies the property `dot(a, ainv) = eye(n)`.

    Arguments
    ---------
    a : asnumpy.ndarray
        Square matrix or batch of square matrices to invert.

    Returns
    -------
    asnumpy.ndarray
        Inverse of the input matrix or matrices.

    Raises
    ------
    LinAlgError
        If the matrix is not square or inversion fails.

    See Also
    --------
    numpy.linalg.inv

    Notes
    -----
    - Computation may be performed on an accelerator device.
    - Inversion promotes dtype to floating point if needed for stability.

    Examples
    --------
    >>> import asnumpy as ap
    >>> M = ap.array([[1., 2.], [3., 4.]])
    >>> ap.linalg.inv(M)
    array([[-2. ,  1. ],
           [ 1.5, -0.5]])
    """
    return ndarray(_inv(a))
