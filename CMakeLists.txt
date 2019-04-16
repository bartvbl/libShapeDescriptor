project(SpinImage CXX CUDA)

cmake_minimum_required (VERSION 3.0)

find_package(OpenMP)
find_package(CUDA REQUIRED)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -lpthread -Wall -std=c++14")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fopenmp /W4 /std:c++latest")
endif()


set(CUDA_NVCC_FLAGS -arch=sm_61 -gencode arch=compute_61,code=\"sm_61,compute_61\" -Xptxas -O3,-v -lcurand -lineinfo -lcudart -std=c++14)





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
                                   lib/lodepng/lodepng.cpp)
file (GLOB_RECURSE TEST_SOURCES    tests/*.cpp)

cuda_add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${PROJECT_NAME} PUBLIC src/)

add_subdirectory(lib/Catch2)
add_executable(runTests ${TEST_SOURCES})
target_link_libraries(runTests SpinImage Catch2::Catch2 cudadevrt)