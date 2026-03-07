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

"""pytest configuration file

This file is the entry point for pytest configuration. It configures the asnumpy test environment:
- Set NumPy weak type promotion rules
- Configure multi-NPU test environment
- Enable pytester plugin for testing asnumpy's testing tools
"""

import logging
import os
import platform
import sys
from typing import List, Optional

import numpy
import pytest

logger = logging.getLogger(__name__)


def _get_npu_device_count() -> int:
    """Get the number of available NPU devices.

    Returns:
        Number of NPU devices, or 0 if detection fails
    """
    try:
        # Import asnumpy to access device detection functions
        # Note: asnumpy.init() is called automatically on import
        import asnumpy

        # Get device count from C++ extension
        count = asnumpy.get_device_count()
        return count
    except Exception as e:
        logger.debug(f"Failed to get NPU device count: {e}")
        return 0


def _validate_npu_id(npu_id: int, available_count: int) -> None:
    """Validate NPU device ID is within valid range.

    Args:
        npu_id: Device ID to validate
        available_count: Total number of available devices

    Raises:
        ValueError: If device ID is invalid
    """
    if available_count == 0:
        raise ValueError(
            f"No NPU devices available, but --npu-id={npu_id} was specified. "
            f"Please check your NPU driver installation."
        )

    if npu_id < 0 or npu_id >= available_count:
        raise ValueError(
            f"Invalid NPU ID {npu_id}. Available NPUs: 0-{available_count - 1}"
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add pytest command-line options.

    Args:
        parser: pytest command-line argument parser
    """
    # Get default values from environment variables
    default_npu_id = int(os.environ.get("ASNUMPY_NPU_ID", "0"))
    default_multi_npu = os.environ.get("ASNUMPY_MULTI_NPU", "").lower() in (
        "1",
        "true",
        "yes",
    )

    parser.addoption(
        "--multi-npu",
        action="store_true",
        default=default_multi_npu,
        help=(
            "Enable multi-NPU testing (requires 2+ NPU devices). "
            "Can also be set via ASNUMPY_MULTI_NPU environment variable."
        ),
    )
    parser.addoption(
        "--npu-id",
        action="store",
        default=default_npu_id,
        type=int,
        metavar="ID",
        help=(
            "Specify the NPU device ID to use (default: 0). "
            "Use --npu-id=1 for the second device, etc. "
            "Can also be set via ASNUMPY_NPU_ID environment variable."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """pytest configuration hook function.

    Performs necessary configuration before tests start:
    1. Validate NumPy version (>= 1.26)
    2. Configure NumPy weak type promotion
    3. Detect and validate NPU devices
    4. Register custom test markers

    Args:
        config: pytest configuration object

    Raises:
        RuntimeError: If NumPy version is insufficient
        ValueError: If NPU configuration is invalid
    """
    logger.info("Configuring Asnumpy test environment")

    # Check NumPy version >= 1.26
    numpy_version = tuple(map(int, numpy.__version__.split(".")[:2]))
    required_version = (1, 26)

    if numpy_version < required_version:
        error_msg = (
            f"Asnumpy requires NumPy >= 1.26.0, but found {numpy.__version__}. "
            f"Please upgrade: pip install 'numpy>=1.26.0'"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    logger.info(f"NumPy {numpy.__version__} detected (>= 1.26 required)")

    # Set NumPy weak type promotion (stable in 1.26+)
    try:
        numpy._set_promotion_state("weak")
        logger.info("NumPy weak promotion state enabled")
    except AttributeError:
        logger.warning("NumPy weak promotion not available in this version")
    except Exception as e:
        logger.warning(f"Failed to set NumPy promotion state: {e}")

    # Detect NPU devices
    npu_count = _get_npu_device_count()
    config._npu_count = npu_count
    config._npu_available = npu_count > 0

    logger.info(f"Detected {npu_count} NPU device(s)")

    # Validate NPU configuration
    npu_id = config.getoption("--npu-id")
    multi_npu = config.getoption("--multi-npu")

    if npu_count > 0:
        try:
            _validate_npu_id(npu_id, npu_count)
            logger.info(f"Using NPU device {npu_id}")
        except ValueError as e:
            logger.error(f"NPU configuration error: {e}")
            raise
    else:
        logger.warning("No NPU devices detected. NPU tests will be skipped.")

    # Warn if multi-NPU is enabled but insufficient devices
    if multi_npu and npu_count < 2:
        logger.warning(
            f"--multi-npu specified but only {npu_count} NPU(s) available. "
            f"Multi-NPU tests will be skipped."
        )

    # Register custom markers
    config.addinivalue_line(
        "markers", "npu: mark test as requiring a NPU device to run"
    )
    config.addinivalue_line(
        "markers", "multi_npu: mark test as requiring multiple NPU devices to run"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running (> 1 second)")
    logger.debug("Registered custom pytest markers: npu, multi_npu, slow")


def pytest_report_header(config: pytest.Config) -> List[str]:
    """Generate test report header with environment information.

    This function returns strings that will be displayed at the beginning
    of the pytest output, providing useful environment information.

    Args:
        config: pytest configuration object

    Returns:
        List of strings to display in the report header
    """
    lines = [
        f"Python: {sys.version.split()[0]} ({platform.platform()})",
        f"NumPy: {numpy.__version__}",
        f"pytest: {pytest.__version__}",
    ]

    # NPU information
    npu_count = getattr(config, "_npu_count", 0)
    if npu_count > 0:
        # Display NPU device IDs
        device_ids = ", ".join(str(i) for i in range(npu_count))
        lines.append(f"NPU devices: {npu_count} (IDs: {device_ids})")

        # Display current configuration
        npu_id = config.getoption("--npu-id")
        multi_npu = config.getoption("--multi-npu")
        lines.append(f"Active NPU ID: {npu_id}")
        lines.append(f"Multi-NPU mode: {'enabled' if multi_npu else 'disabled'}")
    else:
        lines.append("NPU devices: None (NPU tests will be skipped)")

    return lines


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """Modify test items after collection.

    Automatically skip tests based on hardware availability:
    - Skip tests marked with @pytest.mark.npu if no NPU is available
    - Skip tests marked with @pytest.mark.multi_npu if fewer than 2 NPUs available

    Args:
        config: pytest configuration object
        items: List of collected test items
    """
    npu_count = getattr(config, "_npu_count", 0)

    skip_npu = pytest.mark.skip(reason="No NPU device available")
    skip_multi_npu = pytest.mark.skip(
        reason=f"Requires multiple NPUs but only {npu_count} available"
    )

    skipped_npu = 0
    skipped_multi_npu = 0

    for item in items:
        # Check if marked as needing NPU
        if "npu" in item.keywords and npu_count == 0:
            item.add_marker(skip_npu)
            skipped_npu += 1

        # Check if marked as needing multi-NPU
        if "multi_npu" in item.keywords and npu_count < 2:
            item.add_marker(skip_multi_npu)
            skipped_multi_npu += 1

    if skipped_npu > 0:
        logger.info(f"Skipping {skipped_npu} test(s) requiring NPU (no NPU available)")
    if skipped_multi_npu > 0:
        logger.info(
            f"Skipping {skipped_multi_npu} test(s) requiring multi-NPU (insufficient NPUs)"
        )


@pytest.fixture(scope="session")
def multi_npu(request: pytest.FixtureRequest) -> bool:
    """Multi-NPU test fixture.

    Returns True if --multi-npu option is specified on command line,
    False otherwise.

    Args:
        request: pytest fixture request object

    Returns:
        True if multi-NPU mode is enabled, False otherwise
    """
    return request.config.getoption("--multi-npu")


@pytest.fixture(scope="session")
def npu_id(request: pytest.FixtureRequest) -> int:
    """NPU device ID fixture.

    Returns the NPU device ID specified on command line, default is 0.

    Args:
        request: pytest fixture request object

    Returns:
        NPU device ID to use for testing
    """
    return request.config.getoption("--npu-id")


# Enable pytester plugin (for testing the testing tools themselves)
pytest_plugins = ["pytester"]
