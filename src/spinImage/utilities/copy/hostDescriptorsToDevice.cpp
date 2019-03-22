#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include "hostDescriptorsToDevice.h"

template<typename pixelType>
array<pixelType> copyDescriptorsToDevice(const array<pixelType> &hostDescriptors) {
    array<pixelType> deviceDescriptors;

    size_t bufferSize = sizeof(pixelType) * spinImageWidthPixels * spinImageWidthPixels;

    deviceDescriptors.length = spinImageWidthPixels * spinImageWidthPixels;
    checkCudaErrors(cudaMalloc(&deviceDescriptors.content, bufferSize));
    checkCudaErrors(cudaMemcpy(deviceDescriptors.content, hostDescriptors.content, bufferSize, cudaMemcpyHostToDevice));
    return deviceDescriptors;
}

array<quasiSpinImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<quasiSpinImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<quasiSpinImagePixelType>(hostDescriptors);
}

array<spinImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<spinImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<spinImagePixelType>(hostDescriptors);
}
