# *****************************************************************************
# Copyright (c) 2025 ISE Group at Harbin Institute of Technology. All Rights Reserved.
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

from .asnumpy_core.cann import (
    finalize as ap_finalize,
    init as ap_init,
    reset_device as ap_reset_device,
    reset_device_force as ap_reset_device_force,
    set_device as ap_set_device
)

def set_device(device_id: int) -> None:
    return ap_set_device(device_id)

def reset_device(device_id: int) -> None:
    return ap_reset_device(device_id)

def reset_device_force(device_id: int) -> None:
    return ap_reset_device_force(device_id)

def init() -> None:
    return ap_init()

def finalize() -> None:
    return ap_finalize()