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

import sys

import numpy as np

from test_utils import main
import asnumpy as ap


def test_debug_numpy_dtype_str():
    dtype_str = str(ap.dtypes.test_numpy_dtype_str())
    assert "int32" in dtype_str


def test_debug_numpy_create_array():
    np_arr = ap.dtypes.test_numpy_create_array()
    assert isinstance(np_arr, np.ndarray)
    assert np_arr.dtype == np.dtype("int32")
    assert np_arr.shape == (3,)


if __name__ == "__main__":
    tests = [
        test_debug_numpy_dtype_str,
        test_debug_numpy_create_array,
    ]
    sys.exit(main(tests))

