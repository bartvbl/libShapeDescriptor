#pragma once

#include <cassert>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <nvidia/helper_cuda.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct VertexList {
            float* array = nullptr;
            size_t length = 0;

            VertexList(size_t length) {
                checkCudaErrors(cudaMalloc((void**) &array, 3 * length * sizeof(float)));
                this->length = length;
            }

            // For copying
            VertexList() {}

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

            void free() {
                checkCudaErrors(cudaFree(array));
            }
        };
    }
}

