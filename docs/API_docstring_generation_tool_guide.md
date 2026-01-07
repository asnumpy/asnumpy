# API文档自动生成工具使用指南

## 概述

本项目采用[Sphinx](https://sphinxsearch.com/)工具，通过提取项目 Python 代码（如模块、类、函数）中的 `docstring`（文档字符串）自动生成标准化的 Python API 文档。该文档可清晰展示代码的功能说明、参数定义、返回值类型及使用示例，旨在降低文档维护成本，确保代码与文档的一致性。当前已在项目根目录下创建 `docs` 目录，并完成核心配置文件（`conf.py`、`index.rst`、`Makefile` 等）的初始化，用户可直接基于现有框架完成文档的更新与生成。

## 环境准备

以开发者模式安装文档生成工具所需的依赖。

- 开发者模式

    在项目根目录下执行如下命令，安装项目和文档生成工具所需的依赖。

    ```shell
    pip install -e ".[docs]"

    ```
## 文档生成核心文件说明

`docs` 目录下的关键文件功能如下，请勿随意删除核心文件，如需修改请参考下方指引。

| 文件名           | 作用说明                                           |
| ------------- | ---------------------------------------------- |
| `API_docstring_generation_tool_guide.md`     | API文档自动生成工具使用指南      |
| `make.bat`   | Windows端自动化脚本文件                |
| `Makefile`    | 自动化脚本文件，提供 `make html` 等命令用于快速生成文档               |

`docs/source` 目录下的关键文件功能如下，请勿随意删除核心文件，如需修改请参考下方指引。

| 文件名           | 作用说明                                           |
| ------------- | ---------------------------------------------- |
| `conf.py`     | Sphinx 核心配置文件，包含文档主题、扩展插件、项目信息（如名称、版本）等配置      |
| `index.rst`   | 文档首页入口文件，定义文档的目录结构（如模块列表、子文档链接）                |
| `reference/`    | API分类目录文件              |
| `user_guide/`  | 用户指南文件               |

## 标准 API 文档生成流程

首次生成或全量更新文档时，执行以下步骤：

1. **进入 docs 目录**：在项目根目录下打开终端，切换到 `docs` 目录。

    ```shell
    cd docs
    ```

2. **生成文档**：执行 `Makefile` 中的 `html` 命令，Sphinx 会自动提取 `docstring` 并生成 html 静态网页。根据需要执行相应命令。

    ```shell
    # Linux/Mac 环境
    make html       # 生成html格式文档
    # Windows 环境
    make.bat # 生成html格式文档
    ```

4.  **查看生成的文档**：文档生成成功后，会保存在 `docs/build/html` 目录下，直接打开`index.html` 文件，按选项跳转即可查看完整的 API 文档。

## 更新API文档流程

当用户修改了 Python 文件（如新增函数、更新 `docstring` 内容），无需执行API文档生成的全量流程，仅需执行以下简化步骤：

1.  **确认docstring内容**：在项目asnumpy/asnumpy文件夹内合适文件添加python前端声明和注释，确保修改后的函数 / 类 / 模块已按规范编写 `docstring`（推荐 Google 风格或 NumPy 风格，规范细节可参考[Google 风格官方指南](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)、[NumPy 风格官方指南](https://numpydoc.readthedocs.io/en/latest/format.html)），以添加add API为例：
    1. 在asnumpy/asnumpy/__init__.py内声明add
    2. 在asnumpy/asnumpy/math.py内添加add as _ap_add和如下详细声明：
    ```python
    def add(
        x1: Union[ndarray, Any], x2: Union[ndarray, Any], dtype: Optional[np.dtype] = None
    ) -> ndarray:
    """
    Calculate the sum of two inputs element-wise.

    This function adds `x1` and `x2` element by element.

    Arguments
    ---------
    x1 : array-like or scalar
        First input array or scalar.
    x2 : array-like or scalar
        Second input array or scalar.
    dtype : data-type, optional
        Desired data type for the output array.

    Returns
    -------
    asnumpy.ndarray
        The sum of `x1` and `x2`.

    See Also
    --------
    numpy.add
    asnumpy.subtract

    Examples
    --------
    >>> import asnumpy as ap
    >>> ap.add(ap.array([10, 20]), ap.array([5, 5]))
    array([15, 25])
    """
    return ndarray(_ap_add(x1, x2, _convert_dtype(dtype)))
    ```
    3. 在/home/ma-user/work/asnumpy/docs/source/reference/math.rst合适位置添加add
    4. **注 ：（可选）更新模块索引**：若项目新增或删除了 Python 模块，需在asnumpy/docs/source/reference添加对应的rst格式模块索引文件，以及需在asnumpy/docs/source/reference/index.rst中添加模块声明，确保该模块已被包含或已删除。 `rst` 文件格式示例如下。

    ```shell
    模块名称
    =========

    .. currentmodule:: asnumpy

    .. autosummary::
       :toctree: generated/
       :nosignatures:

       API名1
       API名2
    ```

2.  **进入docs目录并重新生成文档**：在 `docs` 目录下执行如下文档生成命令，Sphinx 将自动识别代码变更并更新文档内容。根据需要执行相应命令。

    ```shell
    # Linux/Mac 环境
    cd docs
    make html       # 更新html格式文档
    # Windows 环境
    cd docs 
    make.bat # 更新html格式文档
    ```

3.  **验证文档更新结果**：重新打开 `docs/build/html/index.html`，导航到该修改的模块 / 函数，确认文档内容已同步更新