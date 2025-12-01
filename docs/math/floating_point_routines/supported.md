## floating_point_routines  

**目前已完成的API：** [Signbit](#Signbit)  

<span id="Signbit">1. <mark>**Signbit**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（数值类型）  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入值是否为负数，相当于 `numpy.signbit(x)`，返回与输入形状相同的布尔数组，值为 True 表示该元素符号位为 1（即负数）  
