#pragma once

#include <cassert>
#ifdef CUDACC
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <nvidia/helper_cuda.h>
#else
    struct float3 {
    float x; float y; float z;
    };
#endif
namespace ShapeDescriptor {
    namespace gpu {
        struct VertexList {
            float* array;
            size_t length;
#ifdef CUDACC
            VertexList(size_t length) {
                checkCudaErrors(cudaMalloc((void**) &array, 3 * length * sizeof(float)));
                this->length = length;
            }
#endif

            // For copying
            VertexList() {}
#ifdef CUDACC
            __device__
#endif
            float3 at(size_t index) {
                assert(index < length);

                float3 item;
                item.x = array[index];
                item.y = array[index + length];
                item.z = array[index + 2 * length];
                return item;
            }

#ifdef CUDACC
            __device__
#endif
            void set(size_t index, float3 value) {
                assert(index < length);

                array[index] = value.x;
                array[index + length] = value.y;
                array[index + 2 * length] = value.z;
            }
#ifdef CUDACC
            void free() {
                checkCudaErrors(cudaFree(array));
            }
#endif
        };
    }
}

