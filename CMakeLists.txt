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

cmake_minimum_required(VERSION 3.22)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    fmt
    URL file://${CMAKE_SOURCE_DIR}/third_party/fmt/fmt-12.0.0.tar.gz
)

FetchContent_Declare(
    pybind11
    URL file://${CMAKE_SOURCE_DIR}/third_party/pybind11/pybind11-3.0.1.tar.gz
)

# Disable tests and other unnecessary components for third-party libraries
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fmt pybind11)

project(ascend_numpy)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")

include_directories(${ASCEND_CANN_PACKAGE_PATH}/include)
link_directories(${ASCEND_CANN_PACKAGE_PATH}/lib64)

include_directories(${Python3_NumPy_INCLUDE_DIRS})

include_directories(include)
add_subdirectory(src)
add_subdirectory(python)

# Install Python package files
if(SKBUILD)
    message(STATUS "Building with scikit-build-core")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/asnumpy/"
            DESTINATION "asnumpy"
            FILES_MATCHING
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE)
endif()
