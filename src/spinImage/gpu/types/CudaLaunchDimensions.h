#pragma once

#include <cstddef>

typedef struct CudaLaunchDimensions {
    size_t threadsPerBlock;
    size_t blocksPerGrid;
} CudaLaunchDimensions;

CudaLaunchDimensions calculateCudaLaunchDimensions(size_t vertexCount);