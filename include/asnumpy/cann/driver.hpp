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

#include <acl/acl.h>
#include <cstdint>
#include <vector>

namespace asnumpy {
namespace cann {

/**
 * @brief Initialize the CANN runtime environment
 */
void init();

/**
 * @brief Finalize the CANN runtime environment
 */
void finalize();

/**
 * @brief Get the number of available NPU devices
 * @return Number of NPU devices, or 0 if query fails
 */
int get_device_count();

/**
 * @brief Check if the specified device ID is valid and available
 * @param device_id Device ID to check
 * @return true if device is available, false otherwise
 */
bool is_device_available(int device_id);

}
}