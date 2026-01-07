AsNumpy 基础
============

.. currentmodule:: asnumpy

本节将介绍以下内容：

* :class:`asnumpy.ndarray` 基础
* 当前设备 (Current Device) 的概念
* 主机 (Host) 与设备 (Device) 之间的数组传输

asnumpy.ndarray 基础
--------------------

AsNumpy 是一个旨在昇腾 NPU 上提供与 NumPy 高度兼容的科学计算接口的库。
在下面的代码中，``ap`` 是 ``asnumpy`` 的缩写，遵循将 ``numpy`` 缩写为 ``np`` 的标准惯例：

.. code-block:: python

   >>> import numpy as np
   >>> import asnumpy as ap

:class:`asnumpy.ndarray` 类是 AsNumpy 的核心，它是 :class:`numpy.ndarray` 的替代类。

.. code-block:: python

   >>> x_cpu = np.array([1, 2, 3])
   >>> x_npu = ap.ndarray.from_numpy(x_cpu)

上面的 ``x_npu`` 是 :class:`asnumpy.ndarray` 的一个实例。
:class:`asnumpy.ndarray` 和 :class:`numpy.ndarray` 之间的主要区别在于 AsNumpy 数组是在 *当前设备* 上分配的，我们稍后会讨论这个问题。

大多数数组操作也以类似于 NumPy 的方式完成。
以欧几里得范数（即 L2 范数）为例。
NumPy 有 :func:`numpy.linalg.norm` 函数，可以在 CPU 上计算它。

.. code-block:: python

   >>> x_cpu = np.array([1, 2, 3])
   >>> l2_cpu = np.linalg.norm(x_cpu)

使用 AsNumpy，我们可以以类似的方式在 NPU 上执行相同的计算：

.. code-block:: python

   >>> x_npu = ap.ndarray.from_numpy(np.array([1, 2, 3]))
   >>> l2_npu = ap.linalg.norm(x_npu)

AsNumpy 在 :class:`asnumpy.ndarray` 对象上实现了许多函数。
有关支持的 NumPy API 子集，请参阅 :ref:`API 参考 <asnumpy_reference>`。
NumPy 的知识将帮助您利用 AsNumpy 的大部分功能。
因此，我们建议您熟悉 `NumPy 文档 <https://numpy.org/doc/stable/index.html>`_。

当前设备 (Current Device)
-------------------------

AsNumpy 有一个 *当前设备* 的概念，这是默认的 NPU 设备，数组的分配、操作、计算等都在该设备上进行。
假设当前设备的 ID 为 0。
在这种情况下，以下代码将在 NPU 0 上创建一个数组 ``x_on_npu0``。

.. code-block:: python

   >>> x_on_npu0 = ap.ndarray.from_numpy(np.array([1, 2, 3, 4, 5]))

要切换到另一个 NPU 设备，可以使用 :func:`asnumpy.set_device` 函数：

.. code-block:: python

   >>> ap.set_device(1)
   >>> x_on_npu1 = ap.ndarray.from_numpy(np.array([1, 2, 3, 4, 5]))
   >>> ap.set_device(0)
   >>> x_on_npu0 = ap.ndarray.from_numpy(np.array([1, 2, 3, 4, 5]))

通常，AsNumpy 函数期望数组位于当前设备上。

数据传输 (Data Transfer)
------------------------

将数组移动到设备
~~~~~~~~~~~~~~~~

可以使用 :meth:`asnumpy.ndarray.from_numpy` 将 :class:`numpy.ndarray` 移动到当前设备：

.. code-block:: python

   >>> x_cpu = np.array([1, 2, 3])
   >>> x_npu = ap.ndarray.from_numpy(x_cpu)  # 将数据移动到当前设备

将数组从设备移动到主机
~~~~~~~~~~~~~~~~~~~~~~

可以使用 :meth:`asnumpy.ndarray.to_numpy` 将设备数组移动到主机：

.. code-block:: python

   >>> x_npu = ap.ndarray.from_numpy(np.array([1, 2, 3]))  # 在当前设备中创建数组
   >>> x_cpu = x_npu.to_numpy()  # 将数组移动到主机

内存管理
--------

AsNumpy 使用内存池进行内存管理。