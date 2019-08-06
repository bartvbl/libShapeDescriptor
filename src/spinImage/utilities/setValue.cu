#include "setValue.cuh"
#include <device_launch_parameters.h>

#include <spinImage/libraryBuildSettings.h>

template<typename valueType>
__global__ void setValue(valueType* target, size_t length, valueType value)
{
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < length)
	{
		target[index] = value;
	}
}

// DO NOT REMOVE THIS FUNCTION, COMPILATION WILL FAIL WITHOUT IT
__host__ void dummy()

{
	setValue<spinImagePixelType><<<1, 1, 1>>>(nullptr, 1, 3);
	setValue<quasiSpinImagePixelType><<<1, 1, 1>>>(nullptr, 1, 3);
}