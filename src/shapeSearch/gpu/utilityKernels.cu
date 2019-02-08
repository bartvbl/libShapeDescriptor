#include "utilityKernels.cuh"
#include <device_launch_parameters.h>

#include <shapeSearch/libraryBuildSettings.h>

template<typename valueType>
__global__ void setValue(valueType* target, size_t length, valueType value)
{
	size_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < length)
	{
		target[index] = value;
	}
}

// DO NOTE REMOVE THIS FUNCTION, COMPILATION WILL FAIL WITHOUT IT
__host__ void dummy()

{
	setValue<classicSpinImagePixelType><<<1, 1, 1>>>(nullptr, 1, 3);
	setValue<newSpinImagePixelType><<<1, 1, 1>>>(nullptr, 1, 3);
}