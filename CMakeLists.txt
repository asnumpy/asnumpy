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

cmake_minimum_required(VERSION 3.22...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# CMake policies
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# FetchContent module (for fmt, spdlog, etc.)
include(FetchContent)

# ========== pybind11 (from pip, scikit-build-core auto-configures CMAKE_PREFIX_PATH) ==========
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# ========== Python with NumPy ==========
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# ========== CANN SDK (environment variable lookup) ==========
if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(ASCEND_CANN_PATH "$ENV{ASCEND_TOOLKIT_HOME}")
elseif(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PATH "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_CANN_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(WARNING "CANN SDK path not set. Using default: ${ASCEND_CANN_PATH}. "
                    "Set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH environment variable.")
endif()
message(STATUS "CANN SDK: ${ASCEND_CANN_PATH}")

include_directories(${ASCEND_CANN_PATH}/include)
link_directories(${ASCEND_CANN_PATH}/lib64)

# NumPy include
include_directories(${Python_NumPy_INCLUDE_DIRS})

# ========== fmt (system package first, fallback to FetchContent) ==========
find_package(fmt QUIET)
if(NOT fmt_FOUND)
    message(STATUS "fmt not found in system, fetching from GitHub...")
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.0.0
    )
    set(FMT_TEST OFF CACHE BOOL "" FORCE)
    set(FMT_DOC OFF CACHE BOOL "" FORCE)
    set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(fmt)
else()
    message(STATUS "Using system fmt: ${fmt_VERSION}")
endif()

# ========== spdlog (system package first, fallback to FetchContent) ==========
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found in system, fetching from GitHub...")
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.1
    )
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
    set(SPDLOG_TEST OFF CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
else()
    message(STATUS "Using system spdlog: ${spdlog_VERSION}")
endif()

# ========== Source directories ==========
include_directories(include)
add_subdirectory(src)
add_subdirectory(python)

# ========== Install Python package ==========
if(SKBUILD)
    message(STATUS "Building with scikit-build-core")
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/asnumpy/"
            DESTINATION "asnumpy"
            FILES_MATCHING
            PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE
            PATTERN "*.pyc" EXCLUDE)
endif()
