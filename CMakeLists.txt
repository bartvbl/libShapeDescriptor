project(libShapeSearch CXX)

cmake_minimum_required (VERSION 3.0)

find_package(CUDA REQUIRED)

list(APPEND CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=\"sm_52,compute_52\" -lcurand -lineinfo -lcudart)

include_directories (
        src/
        lib/glm/
        lib/nvidia-samples-common/
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
                                   src/*.c)

cuda_add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})

target_include_directories(${PROJECT_NAME} PUBLIC src/)