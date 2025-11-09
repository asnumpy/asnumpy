## sums_products_differences
**目前已完成的API：** [prod](#prod), [sum](#sum), [nanprod](#nanprod), [nansum](#nansum), [cumprod](#cumprod), [cumsum](#cumsum), [nancumprod](#nancumprod), [nancumsum](#nancumsum), [cross](#cross)
  
<span id="prod">1. <mark> **prod** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，可选，执行乘积运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    keepdims：bool，可选，是否保留运算轴，如果设置为 True，则被缩减的轴将作为大小为一的维度保留在结果中。axis未输入时该项必须未输入，axis输入时该项必须输入  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray（输入axis时）或标量double（未输入axis时）
- **功能：**  
    返回给定轴上数组元素的乘积，如果axis未输入，则返回输入数组中所有元素的乘积

<span id="sum">2. <mark> **sum** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，可选，执行求和运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    keepdims：bool，可选，是否保留运算轴，如果设置为 True，则被缩减的轴将作为大小为一的维度保留在结果中。axis未输入时该项必须未输入，axis输入时该项必须输入  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray（输入axis时）或标量double（未输入axis时）
- **功能：**  
    返回给定轴上数组元素的和，如果axis未输入，则返回输入数组中所有元素的和

<span id="nanprod">3. <mark> **nanprod** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，可选，执行乘积运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    keepdims：bool，可选，是否保留运算轴，如果设置为 True，则被缩减的轴将作为大小为一的维度保留在结果中。axis未输入时该项必须未输入，axis输入时该项必须输入  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray（输入axis时）或标量double（未输入axis时）
- **功能：**  
    返回给定轴上数组元素的乘积，将非数字nan视为 1，如果axis未输入，则返回输入数组中所有元素的乘积

<span id="nansum">4. <mark> **nansum** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，可选，执行求和运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    keepdims：bool，可选，是否保留运算轴，如果设置为 True，则被缩减的轴将作为大小为一的维度保留在结果中。axis未输入时该项必须未输入，axis输入时该项必须输入  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray（输入axis时）或标量double（未输入axis时）
- **功能：**  
    返回给定轴上数组元素的和，将非数字nan视为 0，如果axis未输入，则返回输入数组中所有元素的和

<span id="cumprod">5. <mark> **cumprod** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，执行累积乘积运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray
- **功能：**  
    返回沿给定轴的元素的累积乘积

<span id="cumsum">6. <mark> **cumsum** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，执行累积求和运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray
- **功能：**  
    返回沿给定轴的元素的累积和

<span id="nancumprod">7. <mark> **nancumprod** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，执行累积乘积运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray
- **功能：**  
    返回沿给定轴的元素的累积乘积，将非数字nan视为 1

<span id="nancumsum">8. <mark> **nancumsum** </mark></span>
- **参数：**  
    a：NPUArray，输入值  
    axis：int，执行累积求和运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
    dtype：py::dtype，可选，返回数组的类型，默认使用a的类型
- **返回类型：**  
    NPUArray
- **功能：**  
    返回沿给定轴的元素的累积和，将非数字nan视为 0

<span id="cross">9. <mark> **cross** </mark></span>
- **参数：**  
    a：NPUArray，输入值，a.shape在axis指定的轴广播后的值为3  
    b：NPUArray，输入值，如果 x1.shape != x2.shape，它们必须能够广播到共同的形状  
    axis：int，执行运算的轴，如果 axis 为负，则从最后一个轴倒数到第一个轴  
- **返回类型：**  
    NPUArray
- **功能：**  
    返回 3 元素向量的叉积