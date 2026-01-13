# 文件: src/memory/CMakeLists.txt

# 使用 STATIC 静态库，避免动态库符号不可见的问题
add_library(asnumpy_memory STATIC MemoryPool.cpp)

# 必须开启 PIC，因为这个静态库会被 Python 的动态扩展加载
set_target_properties(asnumpy_memory PROPERTIES POSITION_INDEPENDENT_CODE ON)

# 设置头文件查找路径
target_include_directories(asnumpy_memory PUBLIC ${CMAKE_SOURCE_DIR}/include)

# 链接底层依赖 (CANN SDK)
target_link_libraries(asnumpy_memory PRIVATE ascend_sdk)