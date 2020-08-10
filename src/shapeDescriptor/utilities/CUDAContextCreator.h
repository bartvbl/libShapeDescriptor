#pragma once

#include "cuda_runtime.h"

namespace ShapeDescriptor {
    namespace utilities {
        cudaDeviceProp createCUDAContext(int forceGPU = -1);
        void printGPUProperties(unsigned int deviceIndex);
    }
}
