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
from ._types import ArrayLike, DTypeLike, ShapeLike, ScalarLike
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
def zeros(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros array shape={shape}, dtype={dtype}")
    return ndarray(_zeros(shape, _convert_dtype(dtype)))


@logger.catch
def zeros_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating zeros_like array other={other}, dtype={dtype}")
    return ndarray(_zeros_like(other, _convert_dtype(dtype)))


@logger.catch
def full(shape: ShapeLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full array shape={shape}, value={value}, dtype={dtype}")
    return ndarray(_full(shape, value, _convert_dtype(dtype)))


@logger.catch
def full_like(other: ArrayLike, value: ScalarLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating full_like array other={other}, value={value}, dtype={dtype}")
    return ndarray(_full_like(other, value, _convert_dtype(dtype)))


@logger.catch
def empty(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty array shape={shape}, dtype={dtype}")
    return ndarray(_empty(shape, _convert_dtype(dtype)))


@logger.catch
def empty_like(prototype: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating empty_like array prototype={prototype}, dtype={dtype}")
    return ndarray(_empty_like(prototype, _convert_dtype(dtype)))


@logger.catch
def eye(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating eye array n={n}, dtype={dtype}")
    return ndarray(_eye(n, _convert_dtype(dtype)))


@logger.catch
def ones(shape: ShapeLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones array shape={shape}, dtype={dtype}")
    return ndarray(_ones(shape, _convert_dtype(dtype)))


@logger.catch
def ones_like(other: ArrayLike, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating ones_like array other={other}, dtype={dtype}")
    return ndarray(_ones_like(other, _convert_dtype(dtype)))


@logger.catch
def identity(n: int, dtype: DTypeLike = None) -> ndarray:
    logger.debug(f"Creating identity array n={n}, dtype={dtype}")
    return ndarray(_identity(n, _convert_dtype(dtype)))


@logger.catch
def linspace(
    start: ScalarLike,
    end: ScalarLike,
    steps: int = 50,
    dtype: DTypeLike = None,
) -> ndarray:
    logger.debug(f"Creating linspace array start={start}, end={end}, steps={steps}, dtype={dtype}")
    return ndarray(_linspace(start, end, steps, _convert_dtype(dtype)))
