# *****************************************************************************
# Copyright (c) 2025 AISS Group at Harbin Institute of Technology.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *****************************************************************************

import numpy as np
import asnumpy as ap


def to_npu(array: np.ndarray) -> "ap.ndarray":
    return ap.ndarray.from_numpy(np.array(array))


def print_compare(title: str, asn_result: np.ndarray, np_result: np.ndarray) -> None:
    matches = np.array_equal(asn_result, np_result)
    print(f"{title} match: {matches}")
    if not matches:
        print("asnumpy result:\n", asn_result)
        print("numpy result:\n", np_result)
    print("-" * 60)


def run_argsort_tests() -> None:
    data = np.array([[3, 1, 2], [9, 7, 8]], dtype=np.float32)
    npu_input = to_npu(data)

    expected = np.argsort(data, axis=1)
    asn_indices = ap.argsort(npu_input, axis=1).to_numpy()
    print_compare("Argsort axis=1 ascending", asn_indices, expected)

    expected_desc = np.flip(np.argsort(data, axis=0), axis=0)
    asn_indices_desc = ap.argsort(npu_input, axis=0, descending=True).to_numpy()
    print_compare("Argsort axis=0 descending", asn_indices_desc, expected_desc)


def run_argmax_tests() -> None:
    data = np.array([[1, 5, 2], [9, 3, 4]], dtype=np.float32)
    npu_input = to_npu(data)

    expected = np.argmax(data, axis=1)
    asn_indices = ap.argmax(npu_input, axis=1).to_numpy()
    print_compare("Argmax axis=1", asn_indices, expected)

    expected_keepdim = np.expand_dims(np.argmax(data, axis=-1), axis=-1)
    asn_indices_keepdim = ap.argmax(npu_input, axis=-1, keepdim=True).to_numpy()
    print_compare("Argmax axis=-1 keepdim", asn_indices_keepdim, expected_keepdim)


def run_argmin_tests() -> None:
    data = np.array([[5, 1, 2], [3, 7, 0], [8, 4, 6]], dtype=np.float32)
    npu_input = to_npu(data)

    expected = np.argmin(data, axis=0)
    asn_indices = ap.argmin(npu_input, axis=0).to_numpy()
    print_compare("Argmin axis=0", asn_indices, expected)

    expected_keepdim = np.expand_dims(np.argmin(data, axis=1), axis=1)
    asn_indices_keepdim = ap.argmin(npu_input, axis=1, keepdim=True).to_numpy()
    print_compare("Argmin axis=1 keepdim", asn_indices_keepdim, expected_keepdim)


if __name__ == "__main__":
    print("Running sorting/searching operator checks against NumPy.\n")
    run_argsort_tests()
    run_argmax_tests()
    run_argmin_tests()
    print("Done.")
