include(CheckLanguage)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
	message(STATUS "CUDA support was detected, GPU kernels enabled")
	project(ShapeDescriptor CXX CUDA)
	set(CUDA_NVCC_FLAGS -lcurand -lineinfo -lcudart -lcudadevrt)
	include_directories(
		${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
	)
	enable_language(CUDA)

else()
	message(STATUS "No CUDA support was detected, GPU kernels disabled")
	project(ShapeDescriptor CXX)
endif()

cmake_minimum_required (VERSION 3.5)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)
if(NOT TARGET Catch2)
	add_subdirectory(lib/Catch2-3.8.0)
endif()
add_subdirectory(lib/tinygltf tinygltf)
add_subdirectory(lib/fast-lzma2 fast-lzma2)
add_subdirectory(lib/meshoptimizer meshoptimizer)
find_package(Eigen3 3.4.0 REQUIRED)

add_subdirectory(lib/glm-1.0.1)

include_directories (
	src/
	lib/glm-1.0.1/
	lib/lodepng/
	lib/arrrgh/
	lib/tinygltf
	lib/cuda-samples/Common
	lib/fast-lzma2/src/
	lib/meshoptimizer/src/
	lib/eigen/Eigen
	lib/
	lib/fast-obj
	include/
	${CUDA_INCLUDE_DIRS}
)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -fopenmp -lpthread -Wall")
endif()



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

if (CMAKE_CUDA_COMPILER)
	message("CUDA library created")
	add_library(ShapeDescriptor STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
	target_compile_definitions(ShapeDescriptor PUBLIC DESCRIPTOR_CUDA_KERNELS_ENABLED)
	target_compile_options(ShapeDescriptor PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-DDESCRIPTOR_CUDA_KERNELS_ENABLED>)
	target_include_directories(ShapeDescriptor PUBLIC ${CUDA_INCLUDE_DIRS})
else()
	message("Regular library created")
	add_library(ShapeDescriptor STATIC ${LIBRARY_SOURCES} ${LIBRARY_HEADERS} ${LIBRARY_KERNELS})
	# Ensures .cu files are compiled using the regular C++ compiler
	set_source_files_properties(${LIBRARY_KERNELS} PROPERTIES LANGUAGE CXX)
	set_source_files_properties(${LIBRARY_KERNELS} PROPERTIES COMPILE_OPTIONS -xc++)
endif()
target_include_directories(ShapeDescriptor PUBLIC
		include/ lib/cuda-samples/Common)


target_link_libraries(ShapeDescriptor fast-lzma2 meshoptimizer Eigen3::Eigen)
set_property(TARGET ShapeDescriptor PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 90)

add_executable(runTests ${TEST_SOURCES})
target_link_libraries(runTests ShapeDescriptor Catch2::Catch2WithMain)
set_property(TARGET runTests PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 90)

add_executable(imagerenderer ${RENDER_TOOL_SOURCES})
target_link_libraries(imagerenderer ShapeDescriptor)
set_property(TARGET imagerenderer PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 90)

add_executable(quiccidumper ${QUICCI_DUMPER_TOOL_SOURCES})
target_link_libraries(quiccidumper ShapeDescriptor)
set_property(TARGET quiccidumper PROPERTY CUDA_ARCHITECTURES 52 60 61 70 75 80 90)

add_executable(quicciviewer ${QUICCI_VIEWER_TOOL_SOURCES})
target_link_libraries(quicciviewer ShapeDescriptor)

add_executable(meshconverter tools/meshconverter/main.cpp)
target_link_libraries(meshconverter ShapeDescriptor)