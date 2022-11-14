set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.0)
project(simple_cpu LANGUAGES C CXX)

add_definitions ( -DspinImageWidthPixels=96 )
add_subdirectory(../.. ${CMAKE_CURRENT_BINARY_DIR}/libShapeDescriptor)

include_directories (
        .
        ../../src
)

find_package(OpenMP)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # In case of omp.h missing, run brew install libomp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    include_directories(/usr/local/include /opt/homebrew/include)
endif()

file (GLOB_RECURSE LIBRARY_HEADERS             ./*.h ${OPENMP_INCLUDE_DIRS})
file (GLOB_RECURSE LIBRARY_SOURCES             main.cpp)

add_executable(${PROJECT_NAME} ${LIBRARY_SOURCES} ${LIBRARY_HEADERS})
target_link_libraries(${PROJECT_NAME} ShapeDescriptor fast-lzma2 OpenMP::OpenMP_CXX)
