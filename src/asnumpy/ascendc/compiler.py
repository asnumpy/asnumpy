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

"""Ascend C 编译器接口模块。

负责检测和调用 Ascend C 编译器 (aicc/cce)，管理编译环境和工具链配置。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from loguru import logger


class ArchVersion(Enum):
    """Ascend NPU 架构版本。"""
    ASCEND_910B = "ascend910b"
    ASCEND_910 = "ascend910"
    ASCEND_310P = "ascend310p"


@dataclass
class CompileOptions:
    """Ascend C 编译选项。"""

    arch: ArchVersion = ArchVersion.ASCEND_910B
    optimize: int = 2
    debug: bool = False
    extra_flags: list[str] = field(default_factory=list)
    include_dirs: list[str] = field(default_factory=list)
    defines: dict[str, str] = field(default_factory=dict)


@dataclass
class CompileResult:
    """编译结果。"""

    success: bool
    object_file: Path | None = None
    log: str = ""
    command: str = ""


class AscendCCompilerError(RuntimeError):
    """Ascend C 编译器错误。"""


class AscendCCompiler:
    """Ascend C 编译器管理类。

    负责:
    - 检测 CANN 工具链路径 (ASCEND_TOOLKIT_HOME)
    - 定位 aicc/cce 编译器
    - 编译 Ascend C kernel 源码到目标文件
    - 将目标文件链接为共享库

    Examples:
        compiler = AscendCCompiler()
        print(f"Toolkit: {compiler.toolkit_path}")
        print(f"Compiler: {compiler.compiler_path}")

        result = compiler.compile_source(
            source_code="extern \"C\" __global__ __aicore__ void kernel(...) {}",
            output_name="my_kernel",
        )
        if result.success:
            print(f"Object file: {result.object_file}")
    """

    # 已知的编译器可执行文件名
    _COMPILER_CANDIDATES = ["aicc", "cce", "clang++"]

    def __init__(self, toolkit_path: str | None = None):
        self._toolkit_path = self._detect_toolkit(toolkit_path)
        self._compiler_path = self._detect_compiler()
        self._include_path = self._detect_include_path()
        self._lib_path = self._detect_lib_path()

    # -- 属性 --

    @property
    def toolkit_path(self) -> Path | None:
        return self._toolkit_path

    @property
    def compiler_path(self) -> Path | None:
        return self._compiler_path

    @property
    def include_path(self) -> Path | None:
        return self._include_path

    @property
    def lib_path(self) -> Path | None:
        return self._lib_path

    @property
    def is_available(self) -> bool:
        """编译器是否可用。"""
        return self._compiler_path is not None and self._compiler_path.exists()

    # -- 检测方法 --

    def _detect_toolkit(self, explicit_path: str | None) -> Path | None:
        """检测 CANN 工具链根路径。"""
        if explicit_path:
            p = Path(explicit_path)
            if p.is_dir():
                return p
            raise AscendCCompilerError(f"指定 CANN 路径不存在: {explicit_path}")

        # 检查环境变量
        for env_var in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "ASCEND_CANN_PATH"):
            val = os.getenv(env_var)
            if val:
                p = Path(val)
                if p.is_dir():
                    return p

        # 检查常见安装路径
        candidates = [
            "/usr/local/Ascend/ascend-toolkit/latest",
            "/usr/local/Ascend/latest",
            "/opt/Ascend/ascend-toolkit/latest",
        ]
        for c in candidates:
            p = Path(c)
            if p.is_dir():
                return p

        return None

    def _detect_compiler(self) -> Path | None:
        """检测 Ascend C 编译器可执行文件。"""
        if self._toolkit_path is None:
            return self._find_in_path(self._COMPILER_CANDIDATES)

        # 在 toolkit 路径中搜索
        search_dirs = [
            self._toolkit_path / "compiler" / "bin",
            self._toolkit_path / "tools" / "aicc" / "bin",
            self._toolkit_path / "opp" / "bin",
            self._toolkit_path / "bin",
        ]

        for search_dir in search_dirs:
            for candidate in self._COMPILER_CANDIDATES:
                p = search_dir / candidate
                if p.is_file() and os.access(p, os.X_OK):
                    return p

        # 回退: 在 PATH 中搜索
        return self._find_in_path(self._COMPILER_CANDIDATES)

    def _detect_include_path(self) -> Path | None:
        """检测 CANN 头文件路径。"""
        if self._toolkit_path is None:
            return None

        candidates = [
            self._toolkit_path / "include",
            self._toolkit_path / "acllib" / "include",
            self._toolkit_path / "opp" / "include",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        return None

    def _detect_lib_path(self) -> Path | None:
        """检测 CANN 库文件路径。"""
        if self._toolkit_path is None:
            return None

        candidates = [
            self._toolkit_path / "lib64",
            self._toolkit_path / "acllib" / "lib64",
            self._toolkit_path / "opp" / "lib64",
        ]
        for c in candidates:
            if c.is_dir():
                return c
        return None

    @staticmethod
    def _find_in_path(names: list[str]) -> Path | None:
        """在系统 PATH 中查找可执行文件。"""
        for name in names:
            p = shutil.which(name)
            if p:
                return Path(p)
        return None

    # -- 编译接口 --

    def compile_source(
        self,
        source_code: str,
        output_name: str = "kernel",
        options: CompileOptions | None = None,
        work_dir: str | None = None,
    ) -> CompileResult:
        """编译 Ascend C kernel 源码。

        Args:
            source_code: Ascend C kernel 源码字符串
            output_name: 输出文件名 (不含扩展名)
            options: 编译选项
            work_dir: 工作目录 (None 使用临时目录)

        Returns:
            CompileResult 包含成功状态和输出文件路径
        """
        if options is None:
            options = CompileOptions()

        if not self.is_available:
            raise AscendCCompilerError(
                "Ascend C 编译器不可用。请确保 ASCEND_TOOLKIT_HOME 已设置。"
            )

        # 创建工作目录
        own_dir = False
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ascendc_")
            own_dir = True
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        source_file = work_path / f"{output_name}.cpp"
        object_file = work_path / f"{output_name}.o"

        try:
            source_file.write_text(source_code)
            cmd = self._build_compile_command(source_file, object_file, options)

            logger.info("Compiling: {}", " ".join(str(x) for x in cmd))
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env=self._build_env(),
            )

            log = proc.stdout + "\n" + proc.stderr

            if proc.returncode != 0:
                if own_dir:
                    logger.debug("Build directory kept at: {}", work_path)
                return CompileResult(success=False, log=log, command=" ".join(str(x) for x in cmd))

            logger.info("Compilation successful: {}", object_file)
            return CompileResult(
                success=True,
                object_file=object_file,
                log=log,
                command=" ".join(str(x) for x in cmd),
            )

        except subprocess.TimeoutExpired as e:
            raise AscendCCompilerError("编译超时 (300s)") from e
        except Exception:
            raise

    def link_shared_library(
        self,
        object_files: list[Path],
        output_name: str = "kernel",
        work_dir: str | None = None,
    ) -> CompileResult:
        """将目标文件链接为共享库。

        Args:
            object_files: .o 目标文件列表
            output_name: 输出库名称
            work_dir: 工作目录

        Returns:
            CompileResult
        """
        if work_dir is None:
            work_dir = tempfile.mkdtemp(prefix="ascendc_")
        work_path = Path(work_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        so_file = work_path / f"lib{output_name}.so"

        cmd = [
            str(self._compiler_path),
            "-shared",
            "-fPIC",
            "-o", str(so_file),
            *[str(f) for f in object_files],
        ]

        if self._lib_path:
            cmd.extend(["-L", str(self._lib_path)])
        cmd.extend(["-lascendcl", "-lacl_op_compiler", "-lnnopbase"])

        logger.info("Linking: {}", " ".join(str(x) for x in cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=self._build_env(),
        )

        log = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            return CompileResult(success=False, log=log, command=" ".join(str(x) for x in cmd))

        return CompileResult(
            success=True,
            object_file=so_file,
            log=log,
            command=" ".join(str(x) for x in cmd),
        )

    def _build_compile_command(
        self,
        source_file: Path,
        object_file: Path,
        options: CompileOptions,
    ) -> list[str]:
        """构建编译器命令行。"""
        cmd = [str(self._compiler_path)]

        # 基础标志
        cmd.extend(["-c", str(source_file), "-o", str(object_file)])
        cmd.extend(["-fPIC", "-std=c++17"])

        # 优化级别
        if options.debug:
            cmd.append("-g")
        else:
            cmd.append(f"-O{options.optimize}")

        # 架构
        cmd.append(f"-march={options.arch.value}")

        # 头文件路径
        if self._include_path:
            cmd.extend(["-I", str(self._include_path)])
        for inc in options.include_dirs:
            cmd.extend(["-I", inc])

        # 宏定义
        for k, v in options.defines.items():
            cmd.append(f"-D{k}={v}")

        # 关键 CANN 宏定义
        cmd.extend(["-D__AIBC__", "-D__AICORE__"])

        # 额外标志
        cmd.extend(options.extra_flags)

        return cmd

    def _build_env(self) -> dict:
        """构建编译环境变量。"""
        env = os.environ.copy()
        if self._toolkit_path:
            env["ASCEND_TOOLKIT_HOME"] = str(self._toolkit_path)
            env["ASCEND_CANN_PATH"] = str(self._toolkit_path)
        if self._lib_path:
            ld_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{self._lib_path}:{ld_path}"
        return env

    def get_info(self) -> dict:
        """获取编译器配置摘要。"""
        return {
            "toolkit_path": str(self._toolkit_path) if self._toolkit_path else None,
            "compiler_path": str(self._compiler_path) if self._compiler_path else None,
            "include_path": str(self._include_path) if self._include_path else None,
            "lib_path": str(self._lib_path) if self._lib_path else None,
            "is_available": self.is_available,
        }
