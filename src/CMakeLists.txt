add_library(asnumpy INTERFACE)
add_library(ascend_sdk INTERFACE)

target_include_directories(asnumpy INTERFACE ${CMAKE_SOURCE_DIR}/include)

target_include_directories(ascend_sdk INTERFACE /usr/local/Ascend/ascend-toolkit/latest/include)
target_link_directories(ascend_sdk INTERFACE /usr/local/Ascend/ascend-toolkit/latest/lib64)
target_link_libraries(ascend_sdk INTERFACE ascendcl runtime nnopbase opapi)

add_subdirectory(array)
add_subdirectory(cann)
add_subdirectory(dtypes)
add_subdirectory(linalg)
add_subdirectory(random)
add_subdirectory(math)
add_subdirectory(logic)
add_subdirectory(utils)

target_link_libraries(asnumpy INTERFACE array cann dtypes linalg random math utils)
