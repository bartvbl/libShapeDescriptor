#include "CudaLaunchDimensions.h"

const unsigned int blockSize = 64;

SpinImage::gpu::CudaLaunchDimensions calculateCudaLaunchDimensions(size_t vertexCount)
{
    // Required block count is rounded down, we need rounded up
    size_t blockCount = (vertexCount / blockSize) + 1;
    SpinImage::gpu::CudaLaunchDimensions settings;
    settings.threadsPerBlock = blockSize;
    settings.blocksPerGrid = blockCount;
    return settings;
}