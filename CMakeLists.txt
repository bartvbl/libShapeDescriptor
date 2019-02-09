project(ShapeSearch CXX CUDA)

cmake_minimum_required (VERSION 3.0)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
endif()

find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=\"sm_52,compute_52\" -lcurand -lineinfo -lcudart)

include_directories (
        src/
        lib/glm/
        lib/nvidia-samples-common/
        lib/lodepng/
        ${CUDA_INCLUDE_DIRS}
)

file (GLOB_RECURSE LIBRARY_HEADERS src/*.hpp
                                   src/*.h
                                   src/*.cuh)
file (GLOB_RECURSE LIBRARY_KERNELS src/*.cu
                                   lib/nvidia-samples-common/*.cu)
file (GLOB_RECURSE LIBRARY_SOURCES src/*.cpp
                                   src/*.cxx
                                   src/*.cc
                                   src/*.c
                                   lib/lodepng/*.cpp)

cuda_add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})

target_include_directories(${PROJECT_NAME} PUBLIC src/)