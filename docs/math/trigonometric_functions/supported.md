## trigonometric_functions
**目前已完成的API：** [sin](#sin), [cos](#cos), [tan](#tan), [arcsin](#arcsin), [arccos](#arccos), [arctan](#arctan), [hypot](#hypot), [arctan2](#arctan2), [radians](#radians)  
  
<span id="sin">1. <mark> **sin** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 中每个元素的正弦值 
  
<span id="cos">2. <mark> **cos** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 中每个元素的余弦值 

<span id="tan">3. <mark> **tan** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 中每个元素的正切值 

<span id="arcsin">4. <mark> **arcsin** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    对 x 进行逐元素的反正弦计算

<span id="arccos">5. <mark> **arccos** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    对 x 进行逐元素的反余弦计算

<span id="arctan">6. <mark> **arctan** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    对 x 进行逐元素的反正切计算

<span id="hypot">7. <mark> **hypot** </mark></span>
- **参数：**  
    x1，x2：NPUArray，输入值，三角形的直角边。如果 x1.shape != x2.shape，它们必须能够广播到共同的形状
- **返回类型：**  
    NPUArray
- **功能：**  
    给定直角三角形的“直角边”，返回其斜边，逐元素计算，即sqrt(x1^2 + x2^2)

<span id="arctan2">8. <mark> **arctan2** </mark></span>
- **参数：**  
    x1，x2：NPUArray，y 坐标
    x2：NPUArray，x 坐标，如果 x1.shape != x2.shape，它们必须能够广播到共同的形状  
- **返回类型：**  
    NPUArray
- **功能：**  
    逐元素的 x1/x2 反正切运算

<span id="radians">9. <mark> **radians** </mark></span>
- **参数：**  
    x：NPUArray，以度为单位的输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出对应的弧度值，逐元素计算