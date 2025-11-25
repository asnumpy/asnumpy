## handling_complex_numbers
**目前已完成的API：** [real](#real)  
  
<span id="real">1. <mark> **real** </mark></span>
- **参数：**  
    val：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的实数部分，逐元素计算，如果val是实数，则直接输出该数。如果 val 是实数，则使用 val 的类型作为输出；如果 val 包含复数元素，则返回类型为浮点数。