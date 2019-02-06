#pragma once

#include <cstddef>

template<typename TYPE> struct array
{
	size_t length;
	TYPE* content;
};

typedef array<float> floatArray;
typedef array<int> intArray;
typedef array<unsigned int> uintArray;