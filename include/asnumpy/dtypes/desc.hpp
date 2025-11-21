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

// Prevent multiple inclusion
#pragma once

namespace asnumpy {
namespace dtypes {
    class float8_e5m2;
    template <typename T>
    struct ACLFloatManager;
    template <typename T, typename Enable = void>
    struct TypeDescriptor {
    };
    // 具体类型的特化声明（实现在 reg.cpp 中）
    template<>
    struct TypeDescriptor<float8_e5m2>;
}
}
