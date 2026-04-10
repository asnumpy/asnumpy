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

import numpy as np
import asnumpy as ap  

def check_case(shapes):
    arrays = [ap.ndarray.from_numpy(np.zeros(s, dtype=np.int32)) for s in shapes]

    try:
        result = ap.broadcast_shape(*arrays)
        result_shape = tuple(result)
    except Exception as e:
        result_shape = f"Exception: {e}"

    try:
        expected = np.broadcast_shapes(*shapes)
    except Exception as e:
        expected = f"Exception: {e}"

    print(f"shapes={shapes} -> ap: {result_shape}, numpy: {expected}")

def main():
    test_shapes = [
        [(2, 3), (3,)],          # 一维 + 二维
        [(3,), (3, 1)],          # 不同维度数
        [(3,), ()],              # 标量参与广播
        [(4, 1, 3), (1, 5, 1)],  # 高维广播
        [(2, 3), (2, 3)],        # 相同形状
        [(2, 3), (2, 4)],        # 不可广播
    ]

    for shapes in test_shapes:
        check_case(shapes)

if __name__ == "__main__":
    main()
