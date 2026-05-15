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

"""Ascend C SourceModule — 类似 pycuda.SourceModule 的 JIT 编译接口。

允许用户在 Python 中传入 Ascend C kernel 代码字符串，自动编译为可调用的 NPU kernel。

设计参考:
- pycuda.SourceModule (https://documen.tician.de/pycuda/)
- cupy.RawModule / cupy.RawKernel

核心流程:
    user_code (str) → AscendCCompiler.compile_source() → .o file
    → AscendCCompiler.link_shared_library() → .so file
    → ctypes.CDLL 加载 → 封装为 AscendCKernel 对象

使用示例:
    import asnumpy.ascendc as ac

    kernel_code = '''
    extern "C" __global__ __aicore__ void vec_add(
        __gm__ float* a, __gm__ float* b, __gm__ float* c, int n)
    {
        for (int i = 0; i < n; i++) {
            c[i] = a[i] + b[i];
        }
    }
    '''

    mod = ac.SourceModule(kernel_code, arch="ascend910b")
    vec_add = mod.get_kernel("vec_add")

    a = asnumpy.ones(1024, dtype=asnumpy.float32)
    b = asnumpy.full(1024, 2.0, dtype=asnumpy.float32)
    c = asnumpy.empty(1024, dtype=asnumpy.float32)

    vec_add(a, b, c, 1024, stream=None)
"""

from __future__ import annotations

import ctypes
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from .compiler import AscendCCompiler, AscendCCompilerError, CompileOptions


class AscendCKernel:
    """编译后的 Ascend C kernel 函数包装器。

    封装了通过 ctypes 加载的 kernel 函数指针，提供类型安全的调用接口。
    """

    def __init__(self, name: str, func_ptr: int, module_ref: AscendCSourceModule):
        self._name = name
        self._func_ptr = func_ptr
        self._module = module_ref

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, *args, stream: Any = None, **kwargs):
        """调用 kernel (通过 ACL runtime launch)。

        当前实现: 将参数传递到底层 ACL 调用。

        Args:
            *args: kernel 参数 (NPUArray 指针会被自动提取)
            stream: ACL stream (None 表示默认 stream)
        """
        # 将 NPUArray 参数转换为设备指针
        converted = []
        for arg in args:
            if hasattr(arg, "device_address"):
                converted.append(arg.device_address() or 0)
            elif hasattr(arg, "data_ptr"):
                converted.append(arg.data_ptr() or 0)
            else:
                converted.append(arg)

        logger.debug(
            "Launching kernel '{}' with {} args on stream {}",
            self._name,
            len(converted),
            stream,
        )

        # TODO: 通过 ACL runtime API 启动 kernel
        # 参考 aclopCompileAndExecute / aclLaunchKernel
        # aclopCompileAndExecute(kernel_name, num_inputs, input_descs,
        #                       input_buffers, num_outputs, output_descs,
        #                       output_buffers, attr, stream)
        raise NotImplementedError(
            "Kernel launch via ACL runtime is under development. "
            f"Kernel '{self._name}' compiled successfully but launch is not yet implemented."
        )

    def __repr__(self) -> str:
        return f"AscendCKernel(name={self._name!r})"


class AscendCSourceModule:
    """类似 pycuda.SourceModule 的 Ascend C 源码模块。

    接受 Ascend C kernel 代码字符串，自动编译并加载。

    Examples:
        mod = AscendCSourceModule('''
        extern "C" __global__ __aicore__ void my_kernel(__gm__ float* x, int n) {
            for (int i = 0; i < n; i++) x[i] *= 2.0f;
        }
        ''')
        kernel = mod.get_kernel("my_kernel")
    """

    def __init__(
        self,
        source: str,
        arch: str = "ascend910b",
        options: CompileOptions | None = None,
        toolkit_path: str | None = None,
        no_extern_c: bool = False,
    ):
        """初始化 SourceModule。

        Args:
            source: Ascend C kernel 源码字符串
            arch: 目标架构 (ascend910b, ascend910, ascend310p)
            options: 编译选项
            toolkit_path: CANN 安装路径
            no_extern_c: 如果为 True，自动为源码添加 extern "C" 包装

        Raises:
            AscendCCompilerError: 编译器不可用或编译失败
        """
        self._source = self._preprocess_source(source, no_extern_c)
        self._arch = arch
        self._options = options or CompileOptions()
        self._kernels: dict[str, AscendCKernel] = {}
        self._lib_handle: ctypes.CDLL | None = None
        self._compiler = AscendCCompiler(toolkit_path=toolkit_path)
        self._work_dir = tempfile.mkdtemp(prefix="ascendc_module_")
        self._compiled = False

        logger.info(
            "AscendCSourceModule created: arch={}, compiler_available={}",
            arch,
            self._compiler.is_available,
        )

    # -- 属性 --

    @property
    def source(self) -> str:
        return self._source

    @property
    def arch(self) -> str:
        return self._arch

    @property
    def is_compiled(self) -> bool:
        return self._compiled

    @property
    def kernels(self) -> dict[str, AscendCKernel]:
        return dict(self._kernels)

    # -- 编译与加载 --

    def compile(self) -> AscendCSourceModule:
        """编译源码模块。

        Returns:
            self (支持链式调用)

        Raises:
            AscendCCompilerError: 编译失败
        """
        if self._compiled:
            return self

        if not self._compiler.is_available:
            raise AscendCCompilerError(
                "Ascend C 编译器不可用。请检查 ASCEND_TOOLKIT_HOME 环境变量。"
            )

        module_name = Path(self._work_dir).name

        # Step 1: 编译源码 → .o
        compile_result = self._compiler.compile_source(
            source_code=self._source,
            output_name=module_name,
            options=self._options,
            work_dir=self._work_dir,
        )

        if not compile_result.success:
            raise AscendCCompilerError(
                f"Ascend C 编译失败:\n{compile_result.log}"
            )

        # Step 2: 链接 → .so
        link_result = self._compiler.link_shared_library(
            object_files=[compile_result.object_file],
            output_name=module_name,
            work_dir=self._work_dir,
        )

        if not link_result.success:
            raise AscendCCompilerError(
                f"Ascend C 链接失败:\n{link_result.log}"
            )

        # Step 3: 加载 .so
        so_path = str(link_result.object_file)
        try:
            self._lib_handle = ctypes.CDLL(so_path, mode=ctypes.RTLD_LOCAL)
        except OSError as e:
            raise AscendCCompilerError(
                f"无法加载编译后的共享库 {so_path}: {e}"
            ) from e

        self._compiled = True
        logger.info("AscendCSourceModule compiled and loaded: {}", so_path)
        return self

    def get_kernel(self, name: str) -> AscendCKernel:
        """获取编译后的 kernel 函数。

        Args:
            name: kernel 函数名称

        Returns:
            AscendCKernel 可调用对象

        Raises:
            ValueError: kernel 名称未找到
            AscendCCompilerError: 模块未编译
        """
        if not self._compiled:
            self.compile()

        if name in self._kernels:
            return self._kernels[name]

        try:
            func_ptr = ctypes.cast(
                getattr(self._lib_handle, name),
                ctypes.c_void_p,
            ).value
        except AttributeError as e:
            raise ValueError(
                f"Kernel '{name}' 未在编译后的模块中找到。"
                f"可用的导出符号请使用 nm 或 objdump 检查。"
            ) from e

        kernel = AscendCKernel(name, func_ptr, self)
        self._kernels[name] = kernel
        return kernel

    def get_kernel_names(self) -> list[str]:
        """获取模块中所有 kernel 函数名称 (从源码解析)。"""
        import re

        pattern = r"__global__\s+__aicore__\s+void\s+(\w+)\s*\("
        return re.findall(pattern, self._source)

    # -- 辅助方法 --

    @staticmethod
    def _preprocess_source(source: str, no_extern_c: bool) -> str:
        """预处理源码。"""
        if not no_extern_c:
            header = (
                '#include "kernel_operator.h"\n'
                "using namespace AscendC;\n\n"
            )
            if header not in source:
                source = header + source
        return source

    def get_info(self) -> dict:
        """获取模块配置摘要。"""
        return {
            "arch": self._arch,
            "compiled": self._compiled,
            "kernels": list(self._kernels.keys()),
            "source_size": len(self._source),
            "compiler": self._compiler.get_info(),
        }

    def __repr__(self) -> str:
        status = "compiled" if self._compiled else "source"
        return f"AscendCSourceModule(arch={self._arch}, status={status})"

    def __del__(self):
        """清理临时文件。"""
        import shutil

        if hasattr(self, "_work_dir") and os.path.isdir(self._work_dir):
            try:
                shutil.rmtree(self._work_dir, ignore_errors=True)
            except Exception:
                pass
