add_library(memory STATIC MemoryPool.cpp VirtualMemory.cpp)
set_target_properties(memory PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(memory PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(memory PRIVATE ascend_sdk)
