/******************************************************************************
 * Copyright (c) 2025 AISS Group at Harbin Institute of Technology. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

 #pragma once

// 包含所有浮点类型定义
// 此文件作为统一入口点，保持向后兼容性

#include "float_constants.hpp"
#include "float_utils.hpp"
#include "float8_types.hpp"
#include "bfloat16.hpp"
#include "float6_types.hpp"
#include "float4_types.hpp"
 
 namespace asnumpy {
 namespace dtypes {
    // 所有类型定义已通过上述头文件包含
    // 此命名空间保持向后兼容性
 }  // namespace dtypes
 }  // namespace asnumpy
 