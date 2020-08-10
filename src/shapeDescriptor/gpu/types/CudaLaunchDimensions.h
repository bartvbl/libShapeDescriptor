#pragma once

#include <cstddef>

namespace ShapeDescriptor {
    namespace gpu {
        struct CudaLaunchDimensions {
            size_t threadsPerBlock;
            size_t blocksPerGrid;
        };
    }
}

ShapeDescriptor::gpu::CudaLaunchDimensions calculateCudaLaunchDimensions(size_t vertexCount);