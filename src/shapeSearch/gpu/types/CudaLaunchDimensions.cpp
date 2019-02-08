#include "CudaLaunchDimensions.h"

const unsigned int blockSize = 64;

CudaLaunchDimensions calculateCUDASettings(size_t itemCount, cudaDeviceProp device_information)
{
    size_t blockCount = (itemCount / blockSize) + 1; // Required block count is rounded down, we need rounded up
    CudaLaunchDimensions settings;
    settings.threadsPerBlock = blockSize;
    settings.blocksPerGrid = blockCount;
    return settings;
}