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

import asnumpy as ap
import numpy as np
import time

def main():
    print("Preparing test cases for np.all / ap.all ...")
    
    # 定义多个测试样例
    array_size = (1000, 1000)
    test_cases = [
        np.random.rand(*array_size).astype(bool),
        np.random.rand(*array_size).astype(np.float32),
    ]

    for i, np_arr in enumerate(test_cases):
        print(f"\n=== Test Case {i + 1} ===")

        # 转为 AsNumpy 数组
        ap_arr = ap.ndarray.from_numpy(np_arr)

        # NumPy 执行 all()
        print("Performing all() with NumPy (CPU)...")
        start_np = time.time()
        np_result = np.all(np_arr)
        end_np = time.time()

        # AsNumpy 执行 all()
        print("Performing all() with AsNumpy (NPU/Custom Device)...")
        start_ap = time.time()
        ap_result = ap.all(ap_arr)
        np_ap_result = ap_result.to_numpy() if hasattr(ap_result, "to_numpy") else ap_result
        end_ap = time.time()

        # 对比结果
        print("\n--- Verification ---")
        if np.allclose(np_result, np_ap_result):
            print("✅ SUCCESS: np.all and ap.all produce the same result.")
        else:
            print("❌ FAILURE: np.all and ap.all results differ!")
            print(f"NumPy: {np_result}, AsNumpy: {np_ap_result}")

        # 输出性能信息
        print("\n--- Performance (For Reference) ---")
        print(f"NumPy Time:     {end_np - start_np:.6f} seconds")
        print(f"asnumpy Time:   {end_ap - start_ap:.6f} seconds (Note: Time includes device-host transfer)")

if __name__ == "__main__":
    main()
