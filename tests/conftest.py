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

"""pytest配置文件

这个文件是pytest的入口配置，它告诉pytest如何处理Asnumpy的测试环境：
- 设置NumPy的弱提升规则
- 配置多NPU测试环境
- 启用pytester插件用于测试Asnumpy的测试工具本身
"""

import logging
import numpy
import pytest

logger = logging.getLogger(__name__)


def pytest_configure(config):
    """pytest配置钩子函数
    
    在测试开始前进行必要的配置。
    """
    # 设置NumPy的弱类型提升规则（如果NumPy版本支持）
    try:
        # NumPy 1.20+ 支持弱类型提升
        if hasattr(numpy, '_set_promotion_state'):
            numpy._set_promotion_state('weak')
    except Exception as e:
        # 忽略设置失败，不影响测试运行
        logger.debug("Failed to set numpy promotion state: %s", e)


def pytest_addoption(parser):
    """添加pytest命令行选项
    
    Args:
        parser: pytest的命令行参数解析器
    """
    parser.addoption(
        "--multi-npu",
        action="store_true",
        default=False,
        help="运行多NPU测试"
    )
    parser.addoption(
        "--npu-id",
        action="store",
        default=0,
        type=int,
        help="指定使用的NPU设备ID（默认: 0）"
    )


@pytest.fixture(scope="session")
def multi_npu(request):
    """多NPU测试fixture
    
    如果命令行指定了--multi-npu选项，返回True，否则返回False。
    """
    return request.config.getoption("--multi-npu")


@pytest.fixture(scope="session")
def npu_id(request):
    """NPU设备ID fixture
    
    返回命令行指定的NPU设备ID，默认为0。
    """
    return request.config.getoption("--npu-id")


# 启用pytester插件（用于测试测试工具本身）
pytest_plugins = ["pytester"]

