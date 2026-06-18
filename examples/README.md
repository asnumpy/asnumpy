# AsNumpy项目函数样例说明
样例调用本项目的函数，和Numpy的同功能函数用numpy.allclose进行结果对比，并输出运行时间，以此来展现AsNumpy的准确性和性能

## 已实现样例
| 文件名 | 功能描述 |
| :--- | :-- |
| [01_add](01_add.py) |  用asnumpy.add和numpy.add分别对输入数组 x1 和 x2 执行逐元素加法运算并对比结果，并计算它们的运行时间  |
| [02_exp2](02_exp2.py) |  用asnumpy.exp2和numpy.exp2分别对输入数组 x 的每个元素计算 2 的幂并对比结果，并计算它们的运行时间  |
| [03_multiply](03_multiply.py) |  用asnumpy.multiply和numpy.multiply分别对输入数组 x1 和 x2 执行逐元素乘法运算并对比结果，并计算它们的运行时间  |
| [04_all](04_all.py) |  用asnumpy.all和numpy.all分别对输入数组 x 执行对输入数组执行逻辑与归约操作，判断所有元素是否均为 True并对比结果，并计算它们的运行时间  |
| [05_divide](05_divide.py) |  用asnumpy.divide和numpy.divide分别对输入数组 x1 和 x2 执行逐元素除法并对比结果，并计算它们的运行时间  |
| [06_vdot](06_vdot.py) |  用asnumpy.vdot和numpy.vdot分别计算两个向量的点积并对比结果，并计算它们的运行时间  |
| [07_full](07_full.py) |  用asnumpy.full和numpy.full分别创建填充指定值的数组并对比结果，并计算它们的运行时间  |
| [08_linspace](08_linspace.py) |  用asnumpy.linspace和numpy.linspace分别创建等间距数组并对比结果，并计算它们的运行时间  |
| [09_mean](09_mean.py) |  用asnumpy.mean和numpy.mean分别计算数组均值并对比结果，并计算它们的运行时间  |
| [10_sort](10_sort.py) |  用asnumpy.sort和numpy.sort分别对数组排序并对比结果，并计算它们的运行时间  |
| [11_add_kernel](11_add_kernel.py) |  **Ascend C JIT 编译**：使用 SourceModule 编译自定义向量加法内核，在 NPU 上直接启动执行  |
| [12_custom_kernel](12_custom_kernel.py) |  **Ascend C JIT 多内核**：单次编译包含 vector_mul + scalar_mul 两个内核，演示模板复用  |

## 下一步预期实现样例
| 函数名 | 预期功能描述 |
| :--- | :-- |
| sinh |  用asnumpy.sinh和numpy.sinh分别对输入数组 x1 和 x2 执行逐元素计算双曲正弦并对比结果，并计算它们的运行时间  |
| real |  用asnumpy.real和numpy.real分别逐元素输出 x 的实数部分并对比结果，并计算它们的运行时间  |
| square |  用asnumpy.square和numpy.square分别逐元素计算 x 的平方并对比结果，并计算它们的运行时间  |
| sinc |  用asnumpy.sinc和numpy.sinc分别对输入数组 x 逐元素计算 sinc 函数并对比结果，并计算它们的运行时间  |
| gcd |  用asnumpy.gcd和numpy.gcd分别对输入数组 x1 和 x2 逐元素计算最大公约数并对比结果，并计算它们的运行时间  |
| around |  用asnumpy.around和numpy.around分别逐元素将 x 四舍五入到指定小数位数并对比结果，并计算它们的运行时间  |
| cumsum |  用asnumpy.cumsum和numpy.cumsum分别逐元素计算 x 沿给定轴的元素的累积和并对比结果，并计算它们的运行时间  |
| arcsin |  用asnumpy.arcsin和numpy.arcsin分别对 x 进行逐元素的反正弦计算并对比结果，并计算它们的运行时间  |
| reciprocal |  用asnumpy.reciprocal和numpy.reciprocal分别对 x 计算每个元素的倒数并对比结果，并计算它们的运行时间  |
| binomial |  用asnumpy.binomial从二项分布中抽取足够多随机样本并用卡方分布测试是否符合分布，并计算运行时间  |

## 更新说明
| 时间 | 更新事项 |
| :--- | :-- |
| 2025/10/14 |  新增AsNumpy项目函数样例说明  |
| 2026/03/10 |  优化性能基准测试脚本，适配 NPU 特性：增加预热机制，优化显存管理，采用中段最快速度统计策略
