## rounding  

**目前已完成的API：** [Around](#Around), [Ceil](#Ceil), [Fix](#Fix), [Floor](#Floor), [Rint](#Rint), [Round](#Round), [Trunc](#Trunc)  

<span id="Around">1. <mark>**Around**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    decimals：int，要四舍五入的小数位数；若为负数，则表示在小数点左侧取整  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素将 x 四舍五入到指定小数位数。  

<span id="Ceil">2. <mark>**Ceil**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素返回大于或等于 x 的最小整数（向上取整）。  

<span id="Fix">3. <mark>**Fix**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素向零取整，截取整数部分。  

<span id="Floor">4. <mark>**Floor**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素返回小于或等于 x 的最大整数（向下取整）。  

<span id="Rint">5. <mark>**Rint**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素将 x 四舍五入到最近的整数。  

<span id="Round">6. <mark>**Round**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    decimals：int，要四舍五入的小数位数；若为负数，则表示在小数点左侧取整  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素将 x 四舍五入到指定小数位数（与 Around 功能相同）。  

<span id="Trunc">7. <mark>**Trunc**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    dtype：py::dtype，可选，返回类型，默认与 x 相同  
- **返回类型：**  
    NPUArray  
- **功能：**  
    逐元素向零取整，截断小数部分（与 Fix 功能相同）。
