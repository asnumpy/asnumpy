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

import logging
import sys

logger = logging.getLogger(__name__)


def run_test(func):
    """运行单个测试函数并返回是否通过。
    
    Args:
        func: 要运行的测试函数
        
    Returns:
        bool: 如果测试通过返回 True，否则返回 False
    """
    try:
        func()
        logger.info(f"[PASS] {func.__name__}")
        return True
    except AssertionError as err:
        logger.error(f"[FAIL] {func.__name__}: {err}")
    except Exception as err:
        logger.error(f"[ERROR] {func.__name__}: {err}")
    return False


def run_tests(tests):
    """运行测试列表并输出摘要。
    
    Args:
        tests: 测试函数列表
        
    Returns:
        int: 退出码，0 表示所有测试通过，1 表示有测试失败
    """
    total = len(tests)
    passed = sum(run_test(func) for func in tests)
    logger.info(f"\nSummary: {passed}/{total} tests passed")
    return 0 if passed == total else 1


def main(tests):
    """主函数，运行测试并退出。
    
    Args:
        tests: 测试函数列表
    """
    exit_code = run_tests(tests)
    sys.exit(exit_code)

