## logic  

**目前已完成的API：** [All](#All), [Any](#Any), [IsFinite](#IsFinite), [IsInf](#IsInf), [IsNegInf](#IsNegInf), [IsPosInf](#IsPosInf), [LogicalAnd](#LogicalAnd), [LogicalOr](#LogicalOr), [LogicalNot](#LogicalNot), [LogicalXor](#LogicalXor), [greater](#greater), [greater_equal](#greater_equal), [less](#less), [less_equal](#less_equal), [equal](#equal), [not_equal](#not_equal)  

<span id="All">1. <mark>**All**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（布尔类型）  
    dim：vector\<int64_t\>，可选，指定归约的维度  
    keepdims：bool，可选，是否保留归约的维度，默认为 False  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对输入数组执行逻辑与归约操作，判断所有元素是否均为 True。等价于 numpy.all。  

<span id="Any">2. <mark>**Any**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组（布尔类型）  
    dim：vector\<int64_t\>，可选，指定归约的维度  
    keepdims：bool，可选，是否保留归约的维度，默认为 False  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对输入数组执行逻辑或归约操作，判断是否存在至少一个 True 元素。等价于 numpy.any。  

<span id="IsFinite">3. <mark>**IsFinite**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入是否为有限值（非 inf 且非 nan）。等价于 numpy.isfinite。  

<span id="IsInf">4. <mark>**IsInf**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入是否为正无穷或负无穷。等价于 numpy.isinf。  

<span id="IsNegInf">5. <mark>**IsNegInf**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入是否为负无穷。等价于 numpy.isneginf。  

<span id="IsPosInf">6. <mark>**IsPosInf**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断输入是否为正无穷。等价于 numpy.isposinf。  

<span id="LogicalAnd">7. <mark>**LogicalAnd**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    y：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对应位置执行逻辑与运算。等价于 numpy.logical_and。  

<span id="LogicalOr">8. <mark>**LogicalOr**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    y：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对应位置执行逻辑或运算。等价于 numpy.logical_or。  

<span id="LogicalNot">9. <mark>**LogicalNot**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对每个元素取逻辑非。等价于 numpy.logical_not。  

<span id="LogicalXor">10. <mark>**LogicalXor**</mark></span>  
- **参数：**  
    x：NPUArray，输入数组  
    y：NPUArray，输入数组  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    对应位置执行逻辑异或运算。等价于 numpy.logical_xor。  

<span id="greater">11. <mark>**greater**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray 或标量，第二个输入  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素比较 x1 > x2，返回布尔数组。等价于 numpy.greater。  

<span id="greater_equal">12. <mark>**greater_equal**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray 或标量，第二个输入  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素比较 x1 ≥ x2，返回布尔数组。等价于 numpy.greater_equal。  

<span id="less">13. <mark>**less**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray 或标量，第二个输入  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素比较 x1 < x2，返回布尔数组。等价于 numpy.less。  

<span id="less_equal">14. <mark>**less_equal**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray 或标量，第二个输入  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素比较 x1 ≤ x2，返回布尔数组。等价于 numpy.less_equal。  

<span id="equal">15. <mark>**equal**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray，第二个输入数组  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断 x1 == x2，返回布尔数组。等价于 numpy.equal。  

<span id="not_equal">16. <mark>**not_equal**</mark></span>  
- **参数：**  
    x1：NPUArray，第一个输入数组  
    x2：NPUArray 或标量，第二个输入  
    dtype：py::dtype，可选，返回类型，默认 np.bool_  
- **返回类型：**  
    NPUArray（布尔类型）  
- **功能：**  
    逐元素判断 x1 != x2，返回布尔数组。等价于 numpy.not_equal。  
