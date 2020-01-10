#
# Specify minimum CMake version and project name
#
cmake_minimum_required (VERSION 3.0)

project (fast-lzma2 C)

set (CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set (CMAKE_VERBOSE_MAKEFILE 1) # 1 should be used for debugging
set (CMAKE_SUPPRESS_REGENERATION TRUE) # Suppresses ZERO_CHECK

include_directories (
    src/
)

if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -fPIC -lpthread")
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fopenmp /W4")
endif()

#
# Add files
#
file (GLOB_RECURSE PROJECT_HEADERS 
        src/*.hpp
        src/*.h)
file (GLOB_RECURSE PROJECT_SOURCES 
        src/*.cpp
        src/*.cxx
        src/*.c)
file (GLOB         PROJECT_CONFIGS CMakeLists.txt
        README.rst
        .gitignore
        .gitmodules)

#
# Organizing files
#
source_group ("headers" FILES ${PROJECT_HEADERS})
source_group ("sources" FILES ${PROJECT_SOURCES})

#
# Set executable and target link libraries

add_library (${PROJECT_NAME} STATIC ${PROJECT_SOURCES} ${PROJECT_HEADERS}
        ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
        ${VENDORS_SOURCES})


target_link_libraries (${PROJECT_NAME})