#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace ShapeDescriptor {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;

            __host__ __device__ array() {}

            __host__ array(size_t length) {
                this->length = length;
                checkCudaErrors(cudaMalloc(&content, length * sizeof(TYPE)));
            }

            __host__ __device__ array(size_t length, TYPE* content)
                : length(length),
                  content(content) {}
        };
    }
}