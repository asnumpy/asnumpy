# ******************************************************************************
#  * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
#  *
#  * Licensed under the Apache License, Version 2.0 (the "License");
#  * you may not use this file except in compliance with the License.
#  * You may obtain a copy of the License at
#  *
#  * http://www.apache.org/licenses/LICENSE-2.0
#  *
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS,
#  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  * See the License for the specific language governing permissions and
#  * limitations under the License.
#  ******************************************************************************

pybind11_add_module(asnumpy_core
    asnumpy.cpp
    bind_array.cpp
    bind_cann.cpp
    bind_dtypes.cpp
    bind_fft.cpp
    bind_linalg.cpp
    bind_math.cpp
    bind_logic.cpp
    bind_random.cpp
    bind_sorting.cpp
    bind_testing.cpp
    bind_utils.cpp
    bind_version.cpp)

target_include_directories(asnumpy_core 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(asnumpy_core 
    PUBLIC 
    ascendcl 
    runtime 
    nnopbase 
    opapi
    Python3::NumPy
)
target_link_libraries(asnumpy_core
    PUBLIC
    array
    cann
    dtypes
    utils
    linalg
    random
    math
    logic
    sorting
)

# Install the compiled module
install(TARGETS asnumpy_core
        LIBRARY DESTINATION "asnumpy/lib"
        COMPONENT python)
