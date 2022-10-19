#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>

template<typename valueType>
__global__ void setValue(valueType* target, size_t length, valueType value)
{
    size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        target[index] = value;
    }
}
#endif