#pragma once

// Declaration for use inside the cpu::array header
namespace ShapeDescriptor {
    namespace cpu {
        template<typename TYPE> struct array;
    }
    namespace gpu {
        template<typename TYPE> struct array;
    }
}

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include <shapeDescriptor/utilities/kernels/setValue.cuh>
#include <cstddef>

#include <shapeDescriptor/gpu/gpuCommon.h>

namespace ShapeDescriptor {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length = 0;
            TYPE* content = nullptr;

            __host__ __device__ array() {}

            __host__ array(size_t length) {
                this->length = length;
                cudaMalloc(&content, length * sizeof(TYPE));
            }

            __host__ __device__ array(size_t length, TYPE* content) : length(length), content(content) {}

            __host__ ShapeDescriptor::cpu::array<TYPE> toCPU() {
                return ShapeDescriptor::copy::deviceArrayToHost<TYPE>({length, content});
            }

            __host__ void setValue(TYPE &value) {
                ShapeDescriptor::gpu::setValue<TYPE>(content, length, value);
            }

            __device__ TYPE operator[](size_t index) {
                return content[index];
            }
        };


    }
}