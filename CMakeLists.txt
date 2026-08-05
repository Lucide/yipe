cmake_minimum_required(VERSION 4.4)

# declare C++ and CUDA as native first-class languages
project(yipe LANGUAGES C CXX CUDA)

# declare target language standard specifications cleanly
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    configure_file(
        "${PROJECT_SOURCE_DIR}/.clangd.in"
        "${PROJECT_SOURCE_DIR}/.clangd"
        @ONLY
    )
endif()

add_executable(yipe)

target_sources(yipe
    PRIVATE
    src/main.c
    src/spatial-hash.cu)

target_include_directories(yipe
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

# Enable relocatable device compilation cleanly via target properties
set_target_properties(yipe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native")

target_compile_options(yipe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/wd4514,/wd4820,/wd4668,/wd4459,/wd4626,/wd5027,/wd5045>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda;-Xcudafe;--display_error_number>)

target_link_libraries(yipe PRIVATE CUDA::cudart)
