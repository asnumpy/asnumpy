// Prevent multiple inclusion
#pragma once

// 前向声明，避免循环依赖
namespace asnumpy {
namespace dtypes {

    // 前向声明 ACL 浮点类型
    class float8_e5m2;
    class float8_e4m3fn;
    class float8_e8m0;
    class bfloat16;
    class float6_e2m3fn;
    class float6_e3m2fn;
    class float4_e2m1fn;
    class float4_e1m2fn;
    
    // 前向声明 ACL 整数类型
    class int4;
    class uint1;

    // 前向声明 ACLFloatManager 和 ACLIntManager
    template <typename T>
    struct ACLFloatManager;
    
    template <typename T>
    struct ACLIntManager;

    // TypeDescriptor 模板声明
    template <typename T, typename Enable = void>
    struct TypeDescriptor {
    };

    // 具体类型的特化声明（实现在 reg.cpp 中）
    template<>
    struct TypeDescriptor<float8_e5m2>;

    template<>
    struct TypeDescriptor<float8_e4m3fn>;

    template<>
    struct TypeDescriptor<float8_e8m0>;

    template<>
    struct TypeDescriptor<bfloat16>;

    template<>
    struct TypeDescriptor<float6_e2m3fn>;

    template<>
    struct TypeDescriptor<float6_e3m2fn>;

    template<>
    struct TypeDescriptor<float4_e2m1fn>;

    template<>
    struct TypeDescriptor<float4_e1m2fn>;

    template<>
    struct TypeDescriptor<int4>;

    template<>
    struct TypeDescriptor<uint1>;

}
}
