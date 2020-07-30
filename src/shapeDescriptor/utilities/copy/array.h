#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <nvidia/helper_cuda.h>

namespace SpinImage {
    namespace copy {
        template<typename T>
        SpinImage::cpu::array<T> deviceArrayToHost(SpinImage::gpu::array<T> array) {
            SpinImage::cpu::array<T> hostArray;
            hostArray.length = array.length;
            hostArray.content = new T[array.length];

            checkCudaErrors(cudaMemcpy(hostArray.content, array.content, array.length * sizeof(T), cudaMemcpyDeviceToHost));

            return hostArray;
        }

        template<typename T>
        SpinImage::gpu::array<T> hostArrayToDevice(SpinImage::cpu::array<T> array) {
            SpinImage::gpu::array<T> deviceArray;
            deviceArray.length = array.length;

            size_t bufferSize = sizeof(T) * array.length;

            checkCudaErrors(cudaMalloc((void**) &deviceArray.content, bufferSize));
            checkCudaErrors(cudaMemcpy(deviceArray.content, array.content, bufferSize, cudaMemcpyHostToDevice));

            return deviceArray;
        }
    }
}