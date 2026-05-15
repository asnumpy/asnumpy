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

"""Ascend C JIT 编译模块。

提供类似 pycuda.SourceModule 的接口，支持在 Python 中直接编写、
编译和加载 Ascend C kernel 代码。

核心类:
- SourceModule:   接受 kernel 源码字符串，编译为可调用模块
- AscendCKernel:  封装编译后的 kernel 函数
- AscendCCompiler: 编译器检测与调用接口

Quickstart:
    import asnumpy.ascendc as ac

    mod = ac.SourceModule('''
    extern "C" __global__ __aicore__ void my_kernel(__gm__ float* x, int n) {
        for (int i = 0; i < n; i++) x[i] *= 2.0f;
    }
    ''')
    kernel = mod.get_kernel("my_kernel")
"""

from .compiler import ArchVersion, AscendCCompiler, AscendCCompilerError, CompileOptions
from .module import AscendCKernel, AscendCSourceModule

SourceModule = AscendCSourceModule

__all__ = [
    "SourceModule",
    "AscendCSourceModule",
    "AscendCKernel",
    "AscendCCompiler",
    "AscendCCompilerError",
    "CompileOptions",
    "ArchVersion",
]
