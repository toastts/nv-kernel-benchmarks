cmake_minimum_required(VERSION 3.19)

# don’t use old FindCUDA module (CMP0146)  
cmake_policy(SET CMP0146 NEW)  
# let find_package() use <Foo>_ROOT hints (CMP0074)  
cmake_policy(SET CMP0074 NEW)

project(NVIDIA_SGEMM_PRACTICE LANGUAGES CXX CUDA)

# emit compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# modern C++20 + CUDA14
set(CMAKE_CXX_STANDARD        20   CACHE STRING "")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD       14   CACHE STRING "")
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# target compute capability
set(CMAKE_CUDA_ARCHITECTURES 89)

# point CUDAToolkit_ROOT to /opt/cuda if we need to
# set(CUDAToolkit_ROOT "/opt/cuda" CACHE PATH "CUDA toolkit location")

# pull in cuda headers & libraries, including cudart and cublas
find_package(CUDAToolkit REQUIRED)

# gather all sources (gemm.cu + anything under src/)
file(GLOB_RECURSE NVK_SOURCES
  "${PROJECT_SOURCE_DIR}/src/*.cu"
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
)
list(APPEND NVK_SOURCES "${PROJECT_SOURCE_DIR}/gemm.cu")

# define executable
add_executable(gemm ${NVK_SOURCES})

# in Debug, compile device code with -G for debug symbols
target_compile_options(gemm PRIVATE
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>
)

# headers
target_include_directories(gemm PRIVATE
  ${PROJECT_SOURCE_DIR}/src
)

# link only cuBLAS (cudart comes in transitively)
target_link_libraries(gemm
  PRIVATE
    CUDA::cublas
)
