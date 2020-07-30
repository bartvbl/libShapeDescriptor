#pragma once

#include <cassert>

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "nvidia/helper_cuda.h"

namespace SpinImage {
    namespace gpu {
        struct DeviceVertexList {
            float* array;
            size_t length;

            DeviceVertexList(size_t length) {
                checkCudaErrors(cudaMalloc((void**) &array, 3 * length * sizeof(float)));
                this->length = length;
            }

            // For copying
            DeviceVertexList() {}

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

