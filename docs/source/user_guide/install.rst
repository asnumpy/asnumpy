安装指南
========

使用说明
--------

本仓库目前支持用户以 源代码编译安装 的方式使用，后续将以 whl 包形式提供预编译版本。

环境要求
--------

* 硬件平台：
    * CPU：AArch64或X86_64
    * NPU：昇腾910B
* 系统版本：
    * 主流Linux系统，Ubuntu 20.04及以上版本
* 软件版本：
    * 编译工具：GCC >= 11.2、CMake >= 3.22、ninja-build >= 1.12
    * Python环境：Python >= 3.9、具有pip工具
    * CANN：8.2.RC1.alpha003及以上版本


安装步骤
--------

目前 AsNumpy 处于开发阶段，建议从源码安装。

1. 克隆仓库::

    git clone --recursive https://gitcode.com/cann/asnumpy.git
    cd asnumpy

2. 安装依赖::

    pip install -r requirements.txt

3. 安装 AsNumpy::

    python -m build
    pip install dist/*.whl

验证安装
