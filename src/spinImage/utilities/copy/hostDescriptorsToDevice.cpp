#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include "hostDescriptorsToDevice.h"

template<typename pixelType>
array<pixelType> copyDescriptorsToDevice(const array<pixelType> &hostDescriptors, size_t imageCount) {
    array<pixelType> deviceDescriptors;

    size_t bufferSize = sizeof(pixelType) * spinImageWidthPixels * spinImageWidthPixels * imageCount;

    deviceDescriptors.length = imageCount;
    checkCudaErrors(cudaMalloc(&deviceDescriptors.content, bufferSize));
    checkCudaErrors(cudaMemcpy(deviceDescriptors.content, hostDescriptors.content, bufferSize, cudaMemcpyHostToDevice));
    return deviceDescriptors;
}

array<quasiSpinImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<quasiSpinImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<quasiSpinImagePixelType>(hostDescriptors, imageCount);
}

array<spinImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<spinImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<spinImagePixelType>(hostDescriptors, imageCount);
}
