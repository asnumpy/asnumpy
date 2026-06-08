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

"""锁定 NumPy 2.x 作为项目基线版本。

这些测试在 CI 中起到哨兵作用：如果环境中安装了 NumPy 1.x，
版本检查将直接失败，提示开发者/CI 升级依赖。
"""

import numpy as np


def test_numpy_major_version_is_two_or_newer():
    """确认运行环境中的 NumPy 主版本号 >= 2。"""
    major = int(np.__version__.split(".", 1)[0])
    assert major >= 2, (
        f"NumPy {np.__version__} 不满足基线要求（需要 >= 2.0）。"
        f"请升级：pip install 'numpy>=2.0'"
    )


def test_numpy_result_type_is_the_dtype_oracle():
    """确认 np.result_type 可用作 dtype 提升的参考实现（Oracle）。

    后续任务中 asnumpy 的 dtype promotion 逻辑必须与此行为一致。
    """
    # 混合 int32 + float 标量 → float64
    assert np.result_type(np.array([1], dtype=np.int32), 1.5) == np.dtype("float64")

    # 混合 float32 + float64 标量 → float64（遵循 NumPy 安全提升规则）
    assert np.result_type(np.array([1], dtype=np.float32), np.float64(1)) == np.dtype("float64")
