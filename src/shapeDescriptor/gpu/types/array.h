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
            size_t length = 0;
            TYPE* content = nullptr;

            __host__ __device__ array() {}

            __host__ array(size_t length) {
                this->length = length;
                cudaMalloc(&content, length * sizeof(TYPE));
            }

            __host__ __device__ array(size_t length, TYPE* content) : length(length), content(content) {}

            ShapeDescriptor::cpu::array<TYPE> toCPU() {
                return ShapeDescriptor::copy::deviceArrayToHost<TYPE>({length, content});
            }

            void setValue(TYPE &value) {
                ShapeDescriptor::gpu::setValue<TYPE>(content, length, value);
            }

            TYPE operator[](size_t index) {
                return content[index];
            }
        };


    }
}