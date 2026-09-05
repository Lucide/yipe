cmake_minimum_required(VERSION 4.4)

project(yipe LANGUAGES C CXX CUDA)

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
    # clangd hint
    if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
        set(YIPE_CUDA_ESTIMATED_ARCH "sm_70")
    else()
        set(YIPE_CUDA_ESTIMATED_ARCH "sm_61")
    endif()
    configure_file(
        "${PROJECT_SOURCE_DIR}/.clangd.in"
        "${PROJECT_SOURCE_DIR}/.clangd"
        @ONLY
    )
endif()

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES
    "${PROJECT_SOURCE_DIR}/.clangd"
    "${PROJECT_SOURCE_DIR}/compile_commands.json"
)

add_executable(yipe)

target_sources(yipe
    PRIVATE
    src/main.c
    src/entrypoints.cu)

target_include_directories(yipe
    PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

set_target_properties(yipe PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native")

set(MSVC_OPTIONS -Wall;/Zc:preprocessor;/external:W0;/W4;/permissive-;/wd4514;/wd5045;/wd4668;/wd4626;/wd5027;/wd5031)
list(JOIN MSVC_OPTIONS "," MSVC_OPTIONS_COMMA)

target_compile_options(yipe PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:MSVC>>:${MSVC_OPTIONS}>
    # $<$<COMPILE_LANGUAGE:CUDA>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wsign-conversion;--default-stream per-thread>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=${MSVC_OPTIONS_COMMA}>
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda;-Xptxas=-v,--warn-on-spills,--warn-on-local-memory-usage;-Xcudafe;--display_error_number>)


target_include_directories(yipe SYSTEM PRIVATE
    "${CUDAToolkit_INCLUDE_DIRS}"
)

target_link_libraries(yipe PRIVATE CUDA::cudart)
