#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <nvidia/helper_cuda.h>

namespace ShapeDescriptor {
    namespace copy {
        template<typename T>
        ShapeDescriptor::cpu::array<T> deviceArrayToHost(ShapeDescriptor::gpu::array<T> array) {
            ShapeDescriptor::cpu::array<T> hostArray;
            hostArray.length = array.length;
            hostArray.content = new T[array.length];

            checkCudaErrors(cudaMemcpy(hostArray.content, array.content, array.length * sizeof(T), cudaMemcpyDeviceToHost));

            return hostArray;
        }

        template<typename T>
        ShapeDescriptor::gpu::array<T> hostArrayToDevice(ShapeDescriptor::cpu::array<T> array) {
            ShapeDescriptor::gpu::array<T> deviceArray;
            deviceArray.length = array.length;

            size_t bufferSize = sizeof(T) * array.length;

            checkCudaErrors(cudaMalloc((void**) &deviceArray.content, bufferSize));
            checkCudaErrors(cudaMemcpy(deviceArray.content, array.content, bufferSize, cudaMemcpyHostToDevice));

            return deviceArray;
        }
    }
}