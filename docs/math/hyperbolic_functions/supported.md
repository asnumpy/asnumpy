## hyperbolic_functions  

**目前已完成的API：** [Arccosh](#Arccosh), [Arcsinh](#Arcsinh), [Arctanh](#Arctanh), [Cosh](#Cosh), [Sinh](#Sinh), [Tanh](#Tanh)  

<span id="Arccosh">1. <mark>**Arccosh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（元素需满足 x ≥ 1）  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算反双曲余弦，相当于 `numpy.arccosh(x)`，公式为 `ln(x + √(x² - 1))`。  

<span id="Arcsinh">2. <mark>**Arcsinh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算反双曲正弦，相当于 `numpy.arcsinh(x)`，公式为 `ln(x + √(x² + 1))`。  

<span id="Arctanh">3. <mark>**Arctanh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（元素需满足 |x| < 1）  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算反双曲正切，相当于 `numpy.arctanh(x)`，公式为 `0.5 * ln((1 + x) / (1 - x))`。  

<span id="Cosh">4. <mark>**Cosh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算双曲余弦，相当于 `numpy.cosh(x)`，公式为 `(e^x + e^(-x)) / 2`。  

<span id="Sinh">5. <mark>**Sinh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算双曲正弦，相当于 `numpy.sinh(x)`，公式为 `(e^x - e^(-x)) / 2`。  

<span id="Tanh">6. <mark>**Tanh**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素计算双曲正切，相当于 `numpy.tanh(x)`，公式为 `sinh(x)/cosh(x)`。  
