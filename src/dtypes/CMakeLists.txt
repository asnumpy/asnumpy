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

add_library(dtypes OBJECT 
    reg.cpp
    np_import.cpp
)

target_include_directories(dtypes PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(dtypes PUBLIC 
    fmt::fmt
    ${Python3_LIBRARIES}
)

# 添加必要的编译定义
target_compile_definitions(dtypes PRIVATE
    NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
)

# 确保 C++17 标准
set_target_properties(dtypes PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)