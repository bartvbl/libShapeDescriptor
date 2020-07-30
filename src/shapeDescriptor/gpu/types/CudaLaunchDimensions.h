#pragma once

#include <cstddef>

namespace SpinImage {
    namespace gpu {
        struct CudaLaunchDimensions {
            size_t threadsPerBlock;
            size_t blocksPerGrid;
        };
    }
}

SpinImage::gpu::CudaLaunchDimensions calculateCudaLaunchDimensions(size_t vertexCount);