#pragma once

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <vector_types.h>
#else
struct float2 {
        float x;
        float y;
    };
#endif