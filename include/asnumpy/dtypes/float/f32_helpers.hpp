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

#include <cstdint>
#include <cmath>

namespace asnumpy {
namespace dtypes {

inline void f32_exp_frac_to_unbiased_and_mant(uint32_t exp, uint32_t frac, int& e_unbiased,
                                             float& mant) {
    if (exp == 0) {
        e_unbiased = -126;
        mant = std::ldexp(static_cast<float>(frac), -149);
        return;
    }
    e_unbiased = static_cast<int>(exp) - 127;
    mant = 1.0f + static_cast<float>(frac) * (1.0f / 8388608.0f);
}

inline float mx_decode_sign_exp_mant(uint8_t sign, uint8_t exp, uint8_t mant, int bias,
                                     float mant_step) {
    if (exp == 0) {
        float v = static_cast<float>(mant) * mant_step;
        return sign ? -v : v;
    }
    float base = 1.0f + static_cast<float>(mant) * mant_step;
    int e = static_cast<int>(exp) - bias;
    float v = std::ldexp(base, e);
    return sign ? -v : v;
}

}  // namespace dtypes
}  // namespace asnumpy

