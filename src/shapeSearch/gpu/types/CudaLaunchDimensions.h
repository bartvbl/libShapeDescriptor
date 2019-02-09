#pragma once

#include <cstddef>
#include <driver_types.h>

typedef struct CudaLaunchDimensions {
    size_t threadsPerBlock;
    size_t blocksPerGrid;
} CudaLaunchDimensions;

CudaLaunchDimensions calculateCudaLaunchDimensions(size_t vertexCount, cudaDeviceProp device_information);