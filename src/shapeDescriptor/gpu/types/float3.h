#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <vector_types.h>
#else
    struct float3 {
        float x;
        float y;
        float z;
    };
#endif