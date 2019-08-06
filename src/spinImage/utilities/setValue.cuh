#pragma once

#include <cuda_runtime.h>

template<typename valueType>
__global__ void setValue(valueType* target, size_t length, valueType value);