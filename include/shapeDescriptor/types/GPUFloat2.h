#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#else
struct float2 {
        float x;
        float y;
    };
#endif