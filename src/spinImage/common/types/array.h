#pragma once

#include <cstddef>

template<typename TYPE> struct array
{
	size_t length;
	TYPE* content;
};

typedef array<float> floatArray;