安装指南
========

先决条件
--------

* Python 3.8+
* CANN (Compute Architecture for Neural Networks) 环境

安装步骤
--------

目前 AsNumpy 处于开发阶段，建议从源码安装。

1. 克隆仓库::

    git clone https://github.com/your-repo/asnumpy.git
    cd asnumpy

2. 安装依赖::

    pip install -r requirements.txt

3. 安装 AsNumpy::

    pip install .

验证安装
--------

安装完成后，可以通过以下命令验证是否安装成功::

    python -c "import asnumpy; print(asnumpy.__version__)"