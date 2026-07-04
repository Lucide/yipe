cmake_minimum_required(VERSION 3.18)

# Declare C++ and CUDA as native first-class languages
project(yipe LANGUAGES C CXX CUDA)

# Declare target language standard specifications cleanly
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Configure the executable target with CUDA source files
add_executable(yipe
               src/kernel.cu
               src/main.c
               src/kernel.h)
# Enable relocatable device compilation cleanly via target properties
set_target_properties(yipe PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(yipe PRIVATE CUDA::cudart)
