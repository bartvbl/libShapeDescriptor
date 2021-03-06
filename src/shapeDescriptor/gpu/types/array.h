#pragma once

#include <cstddef>
#ifdef CUDACC
#include <cuda_runtime.h>
#endif

namespace ShapeDescriptor {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;
#ifdef CUDACC
            __host__ __device__
#endif
            array() {}

#ifdef CUDACC
            __host__
#endif
            array(size_t length) {
                this->length = length;
                cudaMalloc(&content, length * sizeof(TYPE));
            }
#ifdef CUDACC
            __host__ __device__
#endif
            array(size_t length, TYPE* content)
                    : length(length),
                      content(content) {}
        };
    }
}