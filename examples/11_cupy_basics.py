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

"""Host/device transfer example based on CuPy's Basics user guide."""

import numpy as np

import asnumpy as ap


def main():
    shape = (1024, 1024)
    x_cpu = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y_cpu = np.full(shape, 2.0, dtype=np.float32)

    # Move host data to the NPU, compute, then copy the result back.
    x_npu = ap.asarray(x_cpu)
    y_npu = ap.array(y_cpu)
    result_cpu = ap.asnumpy(x_npu + y_npu)

    np.testing.assert_array_equal(result_cpu, x_cpu + y_cpu)
    print(f"AsNumpy result: shape={result_cpu.shape}, first row={result_cpu[0, :3]}")
    print("NumPy verification passed.")


if __name__ == "__main__":
    main()
