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

import os
import re
import sys
import subprocess
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

def get_version():
    init_file_path = os.path.join(os.path.dirname(__file__), "asnumpy", "__init__.py")
    with open(init_file_path, "r", encoding="utf-8") as f:
        content = f.read()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.MULTILINE)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string in asnumpy/__init__.py")

class CMakeClean(build_ext):
    def run(self):
        project_root = os.path.abspath(os.path.dirname(__file__))
        build_temp = self.build_temp
        if os.path.exists(build_temp):
            print(f"Cleaning build directory: {build_temp}")
            shutil.rmtree(build_temp)
        super().run()


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        project_root = os.path.abspath(os.path.dirname(__file__))

        build_temp = self.build_temp
        build_lib = self.build_lib
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(os.path.join(build_lib, 'asnumpy', 'lib'))}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={os.path.abspath(os.path.join(build_lib, 'asnumpy', 'lib'))}",
        ]

        try:
            subprocess.check_output(["ninja", "--version"])
            ninja_path = shutil.which("ninja")
            print(f"Found ninja at: {ninja_path}")
            
            cmake_args += [
                "-G", "Ninja",
                f"-DCMAKE_MAKE_PROGRAM={ninja_path}"
            ]
        except (OSError, subprocess.CalledProcessError):
            print("Ninja not found or not working. Falling back to default generator.")
        
        print(f"Configuring CMake in {build_temp}...")
        subprocess.check_call(["cmake", project_root] + cmake_args, cwd=build_temp)

        build_args = ["--config", "Release"]
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            if hasattr(self, "parallel") and self.parallel:
                build_args += [f"-j{self.parallel}"]
            else:
                build_args += [f"-j{os.cpu_count() or 1}"]
        
        print(f"Building with CMake in {build_temp}...")
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

npu_array_extension = Extension(
    name="asnumpy.lib.asnumpy_core",
    sources=[]
)



# Package configuration
setup(
    ext_modules=[npu_array_extension],
    cmdclass={
        "clean": CMakeClean,
        "build_ext": CMakeBuild,
    },
    packages=["asnumpy", "asnumpy.lib"],
    zip_safe = False,
)
