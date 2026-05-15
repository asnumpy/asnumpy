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

"""Ascend C 编译器检测与环境验证测试。

这些测试不依赖 NPU 硬件，仅验证编译器检测逻辑的正确性。
"""

import os
from unittest import mock

import pytest

from asnumpy.ascendc.compiler import (
    ArchVersion,
    AscendCCompiler,
    AscendCCompilerError,
    CompileOptions,
    CompileResult,
)


class TestArchVersion:
    def test_known_architectures(self):
        assert ArchVersion.ASCEND_910B.value == "ascend910b"
        assert ArchVersion.ASCEND_910.value == "ascend910"
        assert ArchVersion.ASCEND_310P.value == "ascend310p"


class TestCompileOptions:
    def test_defaults(self):
        opts = CompileOptions()
        assert opts.arch == ArchVersion.ASCEND_910B
        assert opts.optimize == 2
        assert opts.debug is False
        assert opts.extra_flags == []
        assert opts.include_dirs == []

    def test_custom(self):
        opts = CompileOptions(
            arch=ArchVersion.ASCEND_910,
            optimize=0,
            debug=True,
            extra_flags=["-Wall"],
            include_dirs=["/usr/include"],
            defines={"DEBUG": "1"},
        )
        assert opts.arch == ArchVersion.ASCEND_910
        assert opts.optimize == 0
        assert opts.debug is True
        assert "-Wall" in opts.extra_flags


class TestAscendCCompilerDetection:
    def test_explicit_toolkit_path_raises_if_missing(self):
        with pytest.raises(AscendCCompilerError):
            AscendCCompiler(toolkit_path="/nonexistent/path")

    def test_detect_without_toolkit(self):
        """无 CANN 环境时编译器应优雅降级。"""
        compiler = AscendCCompiler(toolkit_path=None)
        info = compiler.get_info()
        assert isinstance(info, dict)
        assert "toolkit_path" in info
        assert "is_available" in info

    def test_get_info_structure(self):
        compiler = AscendCCompiler(toolkit_path=None)
        info = compiler.get_info()
        for key in ("toolkit_path", "compiler_path", "include_path", "lib_path", "is_available"):
            assert key in info, f"Missing key: {key}"

    def test_is_available_without_toolkit(self):
        compiler = AscendCCompiler(toolkit_path=None)
        # 没有 CANN SDK 时 compiler 应不可用
        assert isinstance(compiler.is_available, bool)


class TestCompileResult:
    def test_success_result(self):
        result = CompileResult(success=True, log="ok", command="aicc -c test.cpp")
        assert result.success is True
        assert result.log == "ok"

    def test_failure_result(self):
        result = CompileResult(success=False, log="error: syntax error", command="aicc -c test.cpp")
        assert result.success is False
        assert "syntax error" in result.log


class TestSourceModule:
    def test_import(self):
        from asnumpy.ascendc import AscendCSourceModule, SourceModule
        assert SourceModule is AscendCSourceModule

    def test_create_module(self):
        from asnumpy.ascendc import AscendCSourceModule

        mod = AscendCSourceModule(
            'extern "C" __global__ __aicore__ void test_kernel(__gm__ float* x, int n) {}',
            arch="ascend910b",
        )
        assert mod.arch == "ascend910b"
        assert mod.is_compiled is False
        assert "test_kernel" in mod.source
        assert "extern" in mod.source

    def test_kernel_name_detection(self):
        from asnumpy.ascendc import AscendCSourceModule

        source = """
        extern "C" __global__ __aicore__ void kernel_a(__gm__ float* x, int n) {}
        extern "C" __global__ __aicore__ void kernel_b(__gm__ float* x, int n) {}
        """
        mod = AscendCSourceModule(source)
        names = mod.get_kernel_names()
        assert "kernel_a" in names
        assert "kernel_b" in names

    def test_compile_without_toolkit_raises(self):
        from asnumpy.ascendc import AscendCSourceModule, AscendCCompilerError

        mod = AscendCSourceModule(
            'extern "C" __global__ __aicore__ void k(__gm__ float* x, int n) {}',
            toolkit_path=None,
        )
        with pytest.raises(AscendCCompilerError):
            mod.compile()

    def test_kernel_repr(self):
        from asnumpy.ascendc import AscendCKernel, AscendCSourceModule

        mod = AscendCSourceModule("")
        kernel = AscendCKernel("my_kernel", 0x1234, mod)
        assert "my_kernel" in repr(kernel)
