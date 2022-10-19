#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "cuda_runtime.h"
#endif

namespace ShapeDescriptor {
    namespace utilities {
        // Returns the ID of the GPU on which the context was created
        int createCUDAContext(int forceGPU = -1);

        void printGPUProperties(unsigned int deviceIndex);
    }
}
