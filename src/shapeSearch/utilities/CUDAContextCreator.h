#pragma once

#include "cuda_runtime.h"

namespace SpinImage {
    namespace utilities {
        cudaDeviceProp createCUDAContext(int forceGPU = -1);
        void printGPUProperties(unsigned int deviceIndex);
    }
}
