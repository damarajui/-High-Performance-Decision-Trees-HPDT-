cmake_minimum_required(VERSION 3.10)
project(PDTO LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDA REQUIRED)

include_directories(include)

add_executable(pdto_cpp
    src/main.cpp
    src/decision_tree.cpp
    src/parallel_cpu.cpp
    src/parallel_gpu.cu
)

set_target_properties(pdto_cpp PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(pdto_cpp ${CUDA_LIBRARIES})

# Add tests
enable_testing()
add_executable(test_decision_tree tests/test_decision_tree.cpp)
add_executable(test_parallel_cpu tests/test_parallel_cpu.cpp)
add_executable(test_parallel_gpu tests/test_parallel_gpu.cu)

add_test(NAME TestDecisionTree COMMAND test_decision_tree)
add_test(NAME TestParallelCPU COMMAND test_parallel_cpu)
add_test(NAME TestParallelGPU COMMAND test_parallel_gpu)