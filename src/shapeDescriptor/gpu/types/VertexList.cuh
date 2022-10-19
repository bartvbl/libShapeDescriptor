#pragma once

#include <cassert>
#include <stdexcept>
#include "shapeDescriptor/libraryBuildSettings.h"

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <nvidia/helper_cuda.h>
#endif

namespace ShapeDescriptor {
    namespace gpu {
        struct VertexList {
            float* array = nullptr;
            size_t length = 0;

            // For copying
            VertexList() {}

            VertexList(size_t length) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
                checkCudaErrors(cudaMalloc((void**) &array, 3 * length * sizeof(float)));
                this->length = length;
#else
                throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
            }
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            __device__ float3 at(size_t index) {
                assert(index < length);

                float3 item;
                item.x = array[index];
                item.y = array[index + length];
                item.z = array[index + 2 * length];
                return item;
            }

            __device__ void set(size_t index, float3 value) {
                assert(index < length);

                array[index] = value.x;
                array[index + length] = value.y;
                array[index + 2 * length] = value.z;
            }
#endif

            void free() {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
                checkCudaErrors(cudaFree(array));
#else
                throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
            }
        };
    }
}

