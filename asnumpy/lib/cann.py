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