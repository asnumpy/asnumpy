# Asnumpy 开发指南

## 目录

1. [简介](#一简介)
2. [后端开发 (C++)](#二后端开发-c)
3. [构建系统配置 (CMake)](#三构建系统配置-cmake)
4. [前端开发 (Python)](#四前端开发-python)
5. [测试编写](#五测试编写)
6. [编译与运行](#六编译与运行)
7. [附录](#七附录)

---

## 一、简介

### 1.1 项目概述

Asnumpy 是一个基于华为昇腾 NPU 的数值计算库，提供与 NumPy 兼容的 API，将计算任务卸载到 NPU 上执行。项目采用分层架构：

- **后端层 (C++)**: 使用昇腾 CANN (Compute Architecture for Neural Networks) API 实现 NPU 算子
- **绑定层 (Pybind11)**: 将 C++ 函数绑定到 Python 接口
- **前端层 (Python)**: 提供用户友好的 API，与 NumPy 接口保持一致
- **测试层**: 使用 pytest 框架和自定义测试工具验证功能正确性

### 1.2 开发流程概览

开发新功能通常遵循以下步骤：

```
1. 添加函数声明 (include/)
   ↓
2. 实现函数逻辑 (src/)
   ↓
3. 配置 CMakeLists.txt (src/)
   ↓
4. 添加 Python 绑定 (python/)
   ↓
5. 添加 Python 包装层 (asnumpy/math.py)
   ↓
6. 导出到主命名空间 (asnumpy/__init__.py)
   ↓
7. 更新 API 文档配置 (docs/source/reference/)
   ↓
8. 编写测试用例 (tests/)
   ↓
9. 编译并运行测试
```

### 1.3 开发环境准备

**必需组件：**
- Python 3.9+
- CMake 3.22+
- 昇腾 CANN 工具包 (8.2.RC1.alpha003+)
- Pybind11
- pytest

**环境变量：**
- `ASCEND_CANN_PACKAGE_PATH`: 指向 CANN 安装目录（默认：`/usr/local/Ascend/ascend-toolkit/latest`）
- `ASCEND_HOME`: 指向昇腾软件安装路径

**编译工具：**
- GCC >= 11.2
- Ninja >= 1.12（推荐，加快编译速度）

---

## 二、后端开发 (C++)

后端开发主要涉及在 C++ 层面实现 NPU 算子调用逻辑。本节以开发 `sinc` 函数为例，详细介绍后端开发流程。

### 2.1 添加函数声明

首先，在对应的头文件中添加函数声明。函数所属的模块决定了头文件的位置。例如，`sinc` 属于数学模块的特殊函数，因此声明位于：

**文件位置**: `include/asnumpy/math/other_special_functions.hpp`

```cpp
/**
 * @brief Compute the normalized sinc function element-wise on the input array.
 *
 * Uses NPU operator aclnnSinc to compute:
 *     sinc(x) = sin(pi * x) / (pi * x), with sinc(0) = 1.
 *
 * @param x Input array.
 * @param dtype Optional output dtype. If not specified, uses input dtype.
 * @return NPUArray Output array with sinc applied element-wise.
 * @throws std::runtime_error If the ACL operator or memory allocation fails.
 */
NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype = std::nullopt);
```

**文档规范：**
- 使用 Doxygen 风格的注释
- `@brief`: 简短描述函数功能
- 详细描述：说明计算公式和特殊值处理
- `@param`: 参数说明
- `@return`: 返回值说明
- `@throws`: 可能抛出的异常类型

### 2.2 实现函数主体

在对应的源文件中实现函数逻辑。继续以 `sinc` 为例：

**文件位置**: `src/math/other_special_functions.cpp`

```cpp
NPUArray Sinc(const NPUArray& x, std::optional<py::dtype> dtype) {
    // 1. 确定输出数据类型
    py::dtype py_dtype = x.dtype;
    aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);
    aclDataType out_dtype = in_dtype;
    
    // 如果指定了 dtype，使用指定的类型
    py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
    if (dtype != std::nullopt) {
        out_py_dtype = *dtype;
        out_dtype = NPUArray::GetACLDataType(out_py_dtype);
    }
    
    // 2. 创建输出数组
    NPUArray out(x.shape, out_py_dtype);

    // 3. 准备 NPU 算子执行所需资源
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 4. 获取 workspace 大小和执行器
    auto error = aclnnSincGetWorkspaceSize(
        x.tensorPtr, out.tensorPtr, &workspaceSize, &executor
    );
    if (error != ACL_SUCCESS) {
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSincGetWorkspaceSize error = "
                          + std::to_string(error);
        const char* detail = aclGetRecentErrMsg();
        if (detail && std::strlen(detail) > 0) msg += " - " + std::string(detail);
        throw std::runtime_error(msg);
    }

    // 5. 分配 workspace 内存（如果需要）
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        error = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (error != ACL_SUCCESS) {
            std::string msg = "[other_special_functions.cpp](sinc) aclrtMalloc error = "
                              + std::to_string(error);
            throw std::runtime_error(msg);
        }
    }

    // 6. 执行算子
    error = aclnnSinc(workspaceAddr, workspaceSize, executor, nullptr);
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclnnSinc error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    // 7. 同步设备，确保计算完成
    error = aclrtSynchronizeDevice();
    if (error != ACL_SUCCESS) {
        if (workspaceAddr) aclrtFree(workspaceAddr);
        std::string msg = "[other_special_functions.cpp](sinc) aclrtSynchronizeDevice error = "
                          + std::to_string(error);
        throw std::runtime_error(msg);
    }

    // 8. 释放 workspace 内存
    if (workspaceAddr) {
        aclrtFree(workspaceAddr);
    }

    return out;
}
```

### 2.3 后端开发规范

#### 2.3.1 错误处理模式

所有 ACL API 调用都需要检查返回值，并提供详细的错误信息：

```cpp
// 标准错误处理模式
auto error = aclnnSomeFunction(/* 参数 */);
if (error != ACL_SUCCESS) {
    std::string msg = "[filename](function) aclnnSomeFunction error = "
                      + std::to_string(error);
    const char* detail = aclGetRecentErrMsg();
    if (detail && std::strlen(detail) > 0) {
        msg += " - " + std::string(detail);
    }
    throw std::runtime_error(msg);
}
```

#### 2.3.2 内存管理

- **Workspace 分配**: 按需分配，及时释放
- **错误路径清理**: 在抛出异常前释放已分配的资源
- **内存分配策略**: 使用 `ACL_MEM_MALLOC_HUGE_FIRST` 优先分配大页内存

#### 2.3.3 数据类型处理

```cpp
// 1. 获取输入数据类型
py::dtype py_dtype = x.dtype;
aclDataType in_dtype = NPUArray::GetACLDataType(py_dtype);

// 2. 确定输出数据类型
aclDataType out_dtype = in_dtype;  // 默认使用输入类型
if (dtype != std::nullopt) {
    out_dtype = NPUArray::GetACLDataType(*dtype);
}

// 3. 创建输出数组
py::dtype out_py_dtype = NPUArray::GetPyDtype(out_dtype);
NPUArray out(x.shape, out_py_dtype);
```

#### 2.3.4 NPU 算子执行流程

典型的 NPU 算子执行流程包括以下步骤：

1. **GetWorkspaceSize**: 获取所需 workspace 大小和执行器
2. **Malloc**: 分配 workspace 内存（如果需要）
3. **Execute**: 执行算子计算
4. **Synchronize**: 同步设备，等待计算完成
5. **Free**: 释放 workspace 内存

---

## 三、构建系统配置 (CMake)

CMake 用于管理项目的编译和链接。添加新功能时需要更新 CMakeLists.txt。

### 3.1 添加新模块

如果要添加一个新的模块（如 `new_module`），需要在 `src/CMakeLists.txt` 中添加：

```cmake
# 添加子目录
add_subdirectory(new_module)

# 链接新模块到主库
target_link_libraries(asnumpy INTERFACE array cann dtypes linalg random math logic sorting statistics nn utils new_module)
```

### 3.2 创建模块的 CMakeLists.txt

在新模块目录下创建 `CMakeLists.txt`：

**文件位置**: `src/new_module/CMakeLists.txt`

```cmake
add_library(new_module OBJECT)

target_sources(new_module PRIVATE
    implementation.cpp
)

target_link_libraries(new_module PUBLIC
    asnumpy
    ascend_sdk
)
```

### 3.3 现有模块添加函数

对于现有模块（如 `math`），只需确保 `src/CMakeLists.txt` 中已包含该模块：

```cmake
add_subdirectory(math)  # math 模块已存在
target_link_libraries(asnumpy INTERFACE ... math ...)  # 已链接
```

如果模块已存在，添加新函数时无需修改 CMakeLists.txt。

---

## 四、前端开发 (Python)

前端开发主要涉及将 C++ 函数暴露到 Python 层，并确保 API 与 NumPy 兼容。

### 4.1 添加 Pybind11 绑定

在对应的绑定文件中添加函数绑定。绑定文件位于 `python/` 目录，按模块组织。

**文件位置**: `python/bind_math.cpp`

```cpp
namespace asnumpy {
    void bind_other_special_functions(py::module_& math);
}

void bind_math(py::module_& math) {
    math.doc() = "math module of asnumpy";
    bind_other_special_functions(math);
}

namespace asnumpy {
    void bind_other_special_functions(py::module_& math){
        math.def("sinc", &Sinc, py::arg("x"), py::arg("dtype") = py::none());
    }
}
```

### 4.2 添加 Python 包装层

在对应的 Python 模块中导入 C++ 函数，并添加 Python 包装层，包括类型注解、文档字符串和参数处理。

**文件位置**: `asnumpy/math.py`

首先，从编译好的 C++ 扩展导入函数：

```python
from .lib.asnumpy_core.math import (
    sin as _ap_sin,
    cos as _ap_cos,
    sinc as _ap_sinc,
    # ... 其他函数
)
from .utils import ndarray, _convert_dtype
```

然后，为每个函数添加 Python 包装层：

```python
def sinc(x: ndarray, dtype: Optional[np.dtype] = None) -> ndarray:
    return ndarray(_ap_sinc(x, _convert_dtype(dtype)))
```

### 4.3 导出到主命名空间

在主包的 `__init__.py` 中添加函数，使其可以通过 `asnumpy.sinc` 访问。

**文件位置**: `asnumpy/__init__.py`

```python
from .math import (
    sin,
    sinc,
    # ... 其他数学函数
)

__all__ = [
    # ... 其他导出
    "sin",
    "sinc",
]
```

### 4.4 前端开发规范

#### 4.4.1 函数命名

- 与 NumPy 保持一致（小写，单词间用下划线分隔）
- 避免使用 Python 关键字
- 保持命名的一致性和可读性

#### 4.4.2 C++ 绑定参数命名

在 Pybind11 绑定中使用清晰的参数名：

```cpp
// 参数命名规范
math.def("sinc", &Sinc, 
         py::arg("x"),                    // 输入数组
         py::arg("dtype") = py::none());  // 可选参数，指定默认值
```

#### 4.4.3 Python 包装层规范

**导入规范：**
```python
# 使用别名导入 C++ 原始函数，使用 _ap_ 前缀标记为内部 API
from .lib.asnumpy_core.math import (
    sin as _ap_sin,
    cos as _ap_cos,
    sinc as _ap_sinc,
)
from .utils import ndarray, _convert_dtype
import numpy as np
from typing import Optional, Union, Sequence, Any
```

**函数定义规范：**
```python
def function_name(
    param1: ndarray, 
    param2: Union[ndarray, Any], 
    optional_param: Optional[np.dtype] = None
) -> ndarray:
    # 参数处理
    if optional_param is not None:
        dtype = _convert_dtype(optional_param)
    
    # 调用 C++ 函数
    return ndarray(_ap_function_name(param1, param2, dtype))
```

---

## 五、测试编写

测试是确保代码质量的关键。asnumpy 使用 pytest 和自定义测试框架编写测试。

### 5.1 测试文件组织

测试文件按模块组织，位于 `tests/asnumpy_tests/` 目录：

```
tests/
├── conftest.py                    # pytest 配置
└── asnumpy_tests/
    ├── math_tests/
    │   ├── test_other_special_functions.py  # 特殊函数测试
    │   ├── test_trigonometric_functions.py
    │   └── ...
    ├── linalg_tests/
    ├── random_tests/
    └── ...
```

### 5.2 编写测试用例

**文件位置**: `tests/asnumpy_tests/math_tests/test_miscellaneous.py`

```python
import numpy
import pytest
from asnumpy import testing

# ========== 辅助函数 ==========

def _create_array(xp, data, dtype):
    """辅助函数：创建数组
    
    解决 asnumpy 尚未实现 xp.array() 接口的问题。
    """
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    # asnumpy 环境
    return xp.ndarray.from_numpy(np_arr)


# ========== 测试用例 ==========

@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_basic(xp, dtype):
    """基础随机测试：测试常规范围内的浮点数"""
    # 必须在函数内部设置固定种子
    # 否则 xp=numpy 和 xp=asnumpy 会生成两组不同的随机数，导致对比失败
    numpy.random.seed(42) 
    
    # 生成 -5 到 5 之间的随机数
    np_a = numpy.random.uniform(low=-5.0, high=5.0, size=(10, 10)).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_sinc_at_zero(xp, dtype):
    """测试 sinc 在 x=0 处的行为"""
    data = [0.0, -0.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_integers(xp, dtype):
    """测试整数点的值"""
    data = [-5.0, -3.0, -1.0, 1.0, 2.0, 4.0]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_halves(xp, dtype):
    """测试半整数点"""
    data = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_special_values(xp, dtype):
    """测试特殊数值：无穷大和 NaN"""
    data = [float('inf'), float('-inf'), float('nan')]
    a = _create_array(xp, data, dtype)
    return xp.sinc(a)


@testing.for_float_dtypes(no_float16=True)
@testing.numpy_asnumpy_allclose(rtol=1e-4, atol=1e-5)
def test_sinc_multidim(xp, dtype):
    """测试多维数组"""
    # 设置随机种子，确保两轮执行数据一致
    numpy.random.seed(123)
    
    np_data = numpy.random.uniform(-2.0, 2.0, size=(2, 3, 4)).astype(dtype)
    a = _create_array(xp, np_data, dtype)
    return xp.sinc(a)
```

### 5.3 测试装饰器

asnumpy 提供了丰富的测试装饰器：

#### 5.3.1 数据类型装饰器

```python
# 测试所有浮点类型
@testing.for_float_dtypes()
def test_something(xp, dtype):
    # dtype 会遍历 float16, float32, float64
    pass

# 排除某些类型
@testing.for_float_dtypes(no_float16=True)
def test_something(xp, dtype):
    # dtype 会遍历 float32, float64
    pass

# 测试整数类型
@testing.for_int_dtypes()
def test_something(xp, dtype):
    # dtype 会遍历 int8, int16, int32, int64, uint8, uint16, uint32, uint64
    pass

# 测试所有类型
@testing.for_all_dtypes()
def test_something(xp, dtype):
    # dtype 会遍历所有支持的类型
    pass
```

#### 5.3.2 NumPy-Asnumpy 比较装饰器

```python
# 比较数组是否完全相等
@testing.numpy_asnumpy_array_equal()
def test_something(xp, dtype):
    data = [1, 2, 3]
    a = _create_array(xp, data, dtype)
    return xp.some_function(a)

# 比较数组是否接近（允许误差）
@testing.numpy_asnumpy_allclose(rtol=1e-5, atol=1e-8)
def test_something(xp, dtype):
    data = [1.0, 2.0, 3.0]
    a = _create_array(xp, data, dtype)
    return xp.some_function(a)
```

#### 5.3.3 参数说明

- `rtol`: 相对容差（relative tolerance）
- `atol`: 绝对容差（absolute tolerance）
- `no_float16`: 排除 float16 类型
- `no_complex`: 排除复数类型

### 5.4 测试用例设计原则

#### 5.4.1 覆盖范围

测试应覆盖以下场景：

1. **基础功能**: 常规输入的预期行为
2. **边界条件**: 极值、零值、特殊值
3. **数据类型**: 支持的多种数据类型
4. **多维数组**: 不同形状的数组
5. **错误处理**: 异常情况的处理

#### 5.4.2 随机数测试

对于涉及随机数的测试，必须设置固定种子：

```python
def test_random_operation(xp, dtype):
    numpy.random.seed(42)  # 确保两轮测试使用相同数据
    np_a = numpy.random.randn(10, 10).astype(dtype)
    a = _create_array(xp, np_a, dtype)
    return xp.some_function(a)
```

#### 5.4.3 辅助函数

使用辅助函数统一数组创建逻辑：

```python
def _create_array(xp, data, dtype):
    """创建数组的辅助函数"""
    np_arr = numpy.array(data, dtype=dtype)
    if xp is numpy:
        return np_arr
    return xp.ndarray.from_numpy(np_arr)
```

---

## 六、编译与运行

### 6.1 编译项目

在开发模式下安装和编译项目，这样修改代码后无需重新安装即可生效：

```bash
pip install -e .
```

**开发模式说明：**
- `-e` 表示 editable 模式，代码修改会立即生效
- 使用 scikit-build-core 构建 C++ 扩展
- 首次安装会自动下载依赖（fmt, pybind11）
- 后续修改 C++ 代码只需重新运行 `pip install -e .`

### 6.2 运行测试

#### 6.2.1 运行所有测试

```bash
pytest tests/
```

#### 6.2.2 运行特定模块测试

```bash
# 数学模块测试
pytest tests/asnumpy_tests/math_tests/

# 线性代数模块测试
pytest tests/asnumpy_tests/linalg_tests/
```

#### 6.2.3 运行单个测试

```bash
# 运行特定测试文件
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py

# 运行特定测试函数
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py::test_sinc_basic
```

### 6.3 调试技巧

#### 6.3.1 查看编译错误

如果编译失败，检查：

1. CMake 输出中的错误信息
2. 确认环境变量是否正确设置
3. 检查头文件路径和库文件路径

#### 6.3.2 查看错误消息

如果 NPU 算子执行失败，错误消息格式为：

```
[filename](function) aclnnXXX error = error_code - detail_message
```

- `filename`: 出错的源文件
- `function`: 出错的函数
- `error_code`: ACL 错误码
- `detail_message`: 详细错误信息（如果有）

常见错误码：
- `ACL_SUCCESS (0)`: 成功
- `ACL_ERROR_INVALID_PARAM (100001)`: 参数错误
- `ACL_ERROR_EXEC_PARAM_ADDR_INVALID (100006)`: 地址无效
- `ACL_ERROR_INVALID_PARAM_NUM (100008)`: 参数数量错误

---

## 七、附录

### A. 项目目录结构

```
asnumpy/
├── asnumpy/                 # Python 包
│   ├── __init__.py         # 主包初始化
│   ├── math.py             # 数学模块
│   ├── linalg/             # 线性代数模块
│   ├── random/             # 随机数模块
│   ├── testing/            # 测试工具
│   └── ...
├── docs/                   # 文档
│   ├── architecture.md     # 架构说明
│   ├── benchmarks.md       # 性能基准
│   ├── developer_guide.md  # 开发指南
│   ├── faq.md              # 常见问题
│   ├── quick_start.md      # 快速入门
│   └── images/             # 文档图片
│       ├── AsNumpy Logo.png
│       ├── NPU扩展功能模块.png
│       └── 功能模块.png
├── include/                # C++ 头文件
│   └── asnumpy/
│       ├── math/           # 数学模块头文件
│       ├── linalg/         # 线性代数模块头文件
│       └── ...
├── src/                    # C++ 源文件
│   ├── math/               # 数学模块实现
│   │   ├── arithmetic_operations.cpp
│   │   ├── trigonometric_functions.cpp
│   │   └── ...
│   ├── linalg/             # 线性代数模块实现
│   └── CMakeLists.txt      # 源文件构建配置
├── python/                 # Pybind11 绑定
│   ├── bind_math.cpp       # 数学模块绑定
│   ├── bind_linalg.cpp     # 线性代数模块绑定
│   └── ...
├── tests/                  # 测试文件
│   ├── conftest.py         # pytest 配置
│   └── asnumpy_tests/      # 测试用例
│       ├── math_tests/
│       ├── linalg_tests/
│       └── ...
├── CMakeLists.txt          # 主 CMake 配置
├── pyproject.toml          # Python 项目配置
└── README.md               # 项目说明
```

### B. 常用命令速查

```bash
# 安装项目
pip install -e .

# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/asnumpy_tests/math_tests/test_miscellaneous.py

# 详细输出
pytest -v

# 显示打印输出
pytest -s

# 清理并重新安装
pip uninstall asnumpy -y && pip install -e .

# 检查 NPU 状态
npu-smi info

# 生成文档
cd docs && make html

# 查看编译输出
pip install -e . -v
```

### C. 参考资源

- [昇腾 CANN 文档](https://www.hiascend.com/document)
- [NumPy 文档](https://numpy.org/doc/stable/)
- [Pybind11 文档](https://pybind11.readthedocs.io/)
- [CMake 文档](https://cmake.org/documentation/)
- [pytest 文档](https://docs.pytest.org/)
- [scikit-build-core 文档](https://scikit-build-core.readthedocs.io/)
