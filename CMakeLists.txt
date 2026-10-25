cmake_minimum_required(VERSION 3.25)
project(cuda_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认 sm_86（RTX 3080 Ti Laptop）；固定架构可让无 GPU 的 CI 机器也能编译
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_library(ops src/vector_add.cu)
target_include_directories(ops PUBLIC include)
target_compile_options(ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
  # Debug 生成设备端调试信息（cuda-gdb）；其他配置保留行号映射（ncu 源码视图）
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-lineinfo>
)

enable_testing()
add_executable(test_vector_add tests/test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE ops)
add_test(NAME vector_add COMMAND test_vector_add)
