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

from loguru import logger
from ._types import ArrayLike, DTypeLike
from .lib.asnumpy_core.array import (
    empty as _empty,
    empty_like as _empty_like,
    eye as _eye,
    full as _full,
    full_like as _full_like,
    identity as _identity,
    linspace as _linspace,
    ones as _ones,
    ones_like as _ones_like,
    zeros as _zeros,
    zeros_like as _zeros_like,
)
from .utils import ndarray, _convert_dtype


@logger.catch
def zeros(shape, dtype: DTypeLike = None, order: str = "C", *, like: ArrayLike = None) -> ndarray:
    logger.debug(f"Creating zeros array shape={shape}, dtype={dtype}")
    return ndarray(_zeros(shape, _convert_dtype(dtype)))


@logger.catch
def zeros_like(
    a: ArrayLike, dtype: DTypeLike = None, order: str = "K", subok: bool = True, shape=None
) -> ndarray:
    logger.debug(f"Creating zeros_like array a={a}, dtype={dtype}")
    return ndarray(_zeros_like(a, _convert_dtype(dtype)))


@logger.catch
def full(
    shape, fill_value, dtype: DTypeLike = None, order: str = "C", *, like: ArrayLike = None
) -> ndarray:
    logger.debug(f"Creating full array shape={shape}, fill_value={fill_value}, dtype={dtype}")
    return ndarray(_full(shape, fill_value, _convert_dtype(dtype)))


@logger.catch
def full_like(
    a: ArrayLike,
    fill_value,
    dtype: DTypeLike = None,
    order: str = "K",
    subok: bool = True,
    shape=None,
) -> ndarray:
    logger.debug(f"Creating full_like array a={a}, fill_value={fill_value}, dtype={dtype}")
    return ndarray(_full_like(a, fill_value, _convert_dtype(dtype)))


@logger.catch
def empty(shape, dtype: DTypeLike = None, order: str = "C", *, like: ArrayLike = None) -> ndarray:
    logger.debug(f"Creating empty array shape={shape}, dtype={dtype}")
    return ndarray(_empty(shape, _convert_dtype(dtype)))


@logger.catch
def empty_like(
    prototype: ArrayLike,
    dtype: DTypeLike = None,
    order: str = "K",
    subok: bool = True,
    shape=None,
    *,
    device: str = None,
) -> ndarray:
    logger.debug(f"Creating empty_like array prototype={prototype}, dtype={dtype}")
    return ndarray(_empty_like(prototype, _convert_dtype(dtype)))


@logger.catch
def eye(
    N: int,
    M: int = None,
    k: int = 0,
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    like: ArrayLike = None,
) -> ndarray:
    logger.debug(f"Creating eye array N={N}, M={M}, k={k}, dtype={dtype}")
    return ndarray(_eye(N, _convert_dtype(dtype)))


@logger.catch
def ones(shape, dtype: DTypeLike = None, order: str = "C", *, like: ArrayLike = None) -> ndarray:
    logger.debug(f"Creating ones array shape={shape}, dtype={dtype}")
    return ndarray(_ones(shape, _convert_dtype(dtype)))


@logger.catch
def ones_like(
    a: ArrayLike, dtype: DTypeLike = None, order: str = "K", subok: bool = True, shape=None
) -> ndarray:
    logger.debug(f"Creating ones_like array a={a}, dtype={dtype}")
    return ndarray(_ones_like(a, _convert_dtype(dtype)))


@logger.catch
def identity(n: int, dtype: DTypeLike = None, *, like: ArrayLike = None) -> ndarray:
    logger.debug(f"Creating identity array n={n}, dtype={dtype}")
    return ndarray(_identity(n, _convert_dtype(dtype)))


@logger.catch
def linspace(
    start,
    stop,
    num: int = 50,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DTypeLike = None,
    axis: int = 0,
) -> ndarray:
    logger.debug(f"Creating linspace array start={start}, stop={stop}, num={num}, dtype={dtype}")
    return ndarray(_linspace(start, stop, num, _convert_dtype(dtype)))
