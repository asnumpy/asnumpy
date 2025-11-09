## arithmetic_operations

**目前已完成的API：** [add](#add), [reciprocal](#reciprocal), [positive](#positive), [negative](#negative), [multiply](#multiply), [divide](#divide), [truedivide](#truedivide), [subtract](#subtract), [floordivide](#floordivide), [power](#power), [float_power](#float_power), [fmod](#fmod), [mod](#mod), [modf](#modf), [remainder](#remainder), [divmod](#divmod)

<span id="add">1. <mark> **add** </mark></span>

- **参数：**
   x1：NPUArray，第一个输入数组
   x2：NPUArray，第二个输入数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   对输入数组 x1 和 x2 执行逐元素加法运算（x1 + x2），支持广播，底层调用 aclnnAdd 算子实现。

<span id="reciprocal">2. <mark> **reciprocal** </mark></span>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   计算每个元素的倒数（1/x），底层调用 aclnnReciprocal 算子。

<span id="positive">3. <mark> **positive** </mark></span>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   一元正操作符。若 dtype 与输入一致则返回副本，否则在 NPU 上执行类型转换（aclnnCast）。

<span id="negative">4. <mark> **negative** </mark></span>

- **参数：**
   x：NPUArray，输入数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   一元负操作符，计算逐元素取负（-x），底层调用 aclnnNeg。

<span id="multiply">5. <mark> **multiply** </mark></span>

- **参数：**
   x1：NPUArray，第一个输入数组
   x2：NPUArray，第二个输入数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   对输入数组执行逐元素乘法运算（x1 * x2），支持广播，底层调用 aclnnMul 算子。

<span id="divide">6. <mark> **divide** </mark></span>

- **参数：**
   x1：NPUArray，被除数数组
   x2：NPUArray，除数数组
   dtype：py::dtype，可选，输出数组的数据类型
- **返回类型：**
   NPUArray
- **功能：**
   执行逐元素除法（x1 / x2），支持广播，底层调用 aclnnDiv 算子。

<span id="truedivide">7. <mark> **truedivide** </mark></span>

- **参数：**
   x1：NPUArray，被除数数组
   x2：NPUArray，除数数组
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   行为等价于 divide，执行逐元素实数除法。

<span id="subtract">8. <mark> **subtract** </mark></span>

- **参数：**
   x1：NPUArray，第一个输入数组
   x2：NPUArray，第二个输入数组
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   执行逐元素减法（x1 - x2），支持广播，底层调用 aclnnSub 算子。

<span id="floordivide">9. <mark> **floordivide** </mark></span>

- **参数：**
   x1：NPUArray，被除数数组
   x2：NPUArray，除数数组
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   计算逐元素向下取整的除法结果 floor(x1 / x2)，底层调用 aclnnFloorDivide 算子。

<span id="power">10. <mark> **power** </mark></span>

- **参数：**
   x1：NPUArray 或标量，底数
   x2：NPUArray 或标量，指数
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   计算 x1 ** x2，支持广播与类型自动转换。底层根据输入类型选择：aclnnPowTensorTensor、aclnnPowScalarTensor 或 aclnnPowTensorScalar。

<span id="float_power">11. <mark> **float_power** </mark></span>

- **参数：**
   x1：NPUArray，底数
   x2：NPUArray，指数
   dtype：py::dtype，可选（必须为浮点类型）
- **返回类型：**
   NPUArray
- **功能：**
   计算浮点幂 x1 ** x2，结果始终为浮点数，底层算子为 aclnnPowTensorTensor。

<span id="fmod">12. <mark> **fmod** </mark></span>

- **参数：**
   x1：NPUArray，被除数
   x2：NPUArray，除数
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   计算逐元素浮点取模：fmod(x1, x2) = x1 - trunc(x1 / x2) * x2，结果符号与被除数一致，底层调用 aclnnFmodTensor。

<span id="mod">13. <mark> **mod** </mark></span>

- **参数：**
   x1：NPUArray，被除数
   x2：NPUArray，除数
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   计算逐元素余数：remainder(x1, x2) = x1 - round(x1 / x2) * x2，与 fmod 不同，结果符号可随除数变化，底层调用 aclnnRemainderTensorTensor。

<span id="modf">14. <mark> **modf** </mark></span>

- **参数：**
   x：NPUArray，输入数组（必须为浮点类型）
- **返回类型：**
   (fractional: NPUArray, integral: NPUArray)
- **功能：**
   将输入拆分为小数部分与整数部分：frac = x - floor(x)，int = floor(x)，底层调用 aclnnFloor 与 aclnnSub。

<span id="remainder">15. <mark> **remainder** </mark></span>

- **参数：**
   x1：NPUArray，被除数
   x2：NPUArray，除数
   dtype：py::dtype，可选
- **返回类型：**
   NPUArray
- **功能：**
   与 mod 等价，返回 x1 - round(x1 / x2) * x2，内部调用 Mod() 实现。

<span id="divmod">16. <mark> **divmod** </mark></span>

- **参数：**
   x1：NPUArray，被除数
   x2：NPUArray，除数
   dtype：py::dtype，可选
- **返回类型：**
   (quotient: NPUArray, remainder: NPUArray)
- **功能：**
   同时计算商与余数：quotient = floor(x1 / x2)，remainder = x1 - quotient * x2，底层调用 aclnnDivMod（mode=2）并手动计算余数。