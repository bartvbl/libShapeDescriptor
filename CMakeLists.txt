project(ShapeDescriptor CXX CUDA)

cmake_minimum_required (VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP)
find_package(CUDA REQUIRED)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -lpthread -lstdc++fs -Wall -std=c++14")
endif()

if (MSVC)
    add_definitions(-D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
endif()



set(CUDA_NVCC_FLAGS -arch=sm_52 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_75,code=compute_75 -lcurand -lineinfo -lcudart -std=c++14)

add_subdirectory(lib/fast-lzma2 ${CMAKE_CURRENT_BINARY_DIR}/fast-lzma2)

include_directories (
        src/
        lib/glm/
        lib/nvidia-samples-common/
        lib/lodepng/
        lib/arrrgh/
        lib/json/
        lib/fast-lzma2/src/
        lib/
        lib/fast-obj
        ${CUDA_INCLUDE_DIRS}
)

file (GLOB_RECURSE LIBRARY_HEADERS             src/*.hpp
                                               src/*.h
                                               src/*.cuh)
file (GLOB_RECURSE LIBRARY_KERNELS             src/*.cu
                                               lib/nvidia-samples-common/*.cu)
file (GLOB_RECURSE LIBRARY_SOURCES             src/*.cpp
                                               src/*.cxx
                                               src/*.cc
                                               src/*.c
                                               lib/lodepng/lodepng.cpp
                                               lib/fast-obj/fast_obj.cpp)
file (GLOB_RECURSE TEST_SOURCES                tests/*.cpp)
file (GLOB_RECURSE RENDER_TOOL_SOURCES         tools/imagerenderer/*.cpp)
file (GLOB_RECURSE QUICCI_DUMPER_TOOL_SOURCES  tools/quiccidumper/*.cpp)
file (GLOB_RECURSE QUICCI_VIEWER_TOOL_SOURCES  tools/quicciviewer/*.cpp)

cuda_add_library(${PROJECT_NAME} STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})

if(MSVC)
	target_link_libraries(${PROJECT_NAME} fast-lzma2)
else()
	target_link_libraries(${PROJECT_NAME} stdc++fs fast-lzma2)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC src/)

add_subdirectory(lib/Catch2)
add_executable(runTests ${TEST_SOURCES})
target_link_libraries(runTests ShapeDescriptor Catch2::Catch2 cudadevrt)

add_executable(imagerenderer ${RENDER_TOOL_SOURCES})
target_link_libraries(imagerenderer ShapeDescriptor cudadevrt)

add_executable(quiccidumper ${QUICCI_DUMPER_TOOL_SOURCES})
target_link_libraries(quiccidumper ShapeDescriptor cudadevrt)

add_executable(quicciviewer ${QUICCI_VIEWER_TOOL_SOURCES})
target_link_libraries(quicciviewer ShapeDescriptor cudadevrt)

if(MSVC)
	set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)
	set_property(TARGET runTests PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)
	set_property(TARGET imagerenderer PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)
	set_property(TARGET quiccidumper PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75)
endif()