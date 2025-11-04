## exponents_and_logarithms
**目前已完成的API：** [exp](#exp), [expm1](#expm1), [exp2](#exp2), [log](#log), [log10](#log10), [log2](#log2), [log1p](#log1p), [logaddexp](#logaddexp), [logaddexp2](#logaddexp2)  
  
<span id="exp">1. <mark> **exp** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的逐元素 e 指数  
  
<span id="expm1">2. <mark> **expm1** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的逐元素 e 指数减 1 ，即exp(x) - 1

<span id="exp2">3. <mark> **exp2** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    对 x 的每个元素计算 2 的幂

<span id="log">4. <mark> **log** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的逐元素自然对数

<span id="log10">5. <mark> **log10** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的以 10 为底的对数，逐元素计算

<span id="log2">6. <mark> **log2** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的以 2 为底的对数，逐元素计算

<span id="log1p">7. <mark> **log1p** </mark></span>
- **参数：**  
    x：NPUArray，输入值
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 x 的加 1 后的自然的对数，逐元素计算，即log(1 + x)

<span id="logaddexp">8. <mark> **logaddexp** </mark></span>
- **参数：**  
    x1，x2：NPUArray，输入值，如果 x1.shape != x2.shape，它们必须能够广播到共同的形状  
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 e 指数和的对数，逐元素计算，即log(exp(x1) + exp(x2))

<span id="logaddexp2">9. <mark> **logaddexp2** </mark></span>
- **参数：**  
    x1，x2：NPUArray，输入值，如果 x1.shape != x2.shape，它们必须能够广播到共同的形状  
- **返回类型：**  
    NPUArray
- **功能：**  
    输出 2 的幂之和的以 2 为底的对数，逐元素计算，即log2(exp2(x1) + exp2(x2))