#pragma once

#include <cstddef>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#endif

namespace ShapeDescriptor {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;

            __host__ __device__ array() {}

            __host__ array(size_t length) {
                this->length = length;
                cudaMalloc(&content, length * sizeof(TYPE));
            }

            __host__ __device__ array(size_t length, TYPE* content)
                : length(length),
                  content(content) {}
        };
    }
}