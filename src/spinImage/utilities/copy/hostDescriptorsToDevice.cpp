#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include "hostDescriptorsToDevice.h"

template<typename pixelType>
SpinImage::array<pixelType> copyDescriptorsToDevice(const SpinImage::array<pixelType> &hostDescriptors, size_t imageCount) {
    SpinImage::array<pixelType> deviceDescriptors;

    size_t bufferSize = sizeof(pixelType) * spinImageWidthPixels * spinImageWidthPixels * imageCount;

    deviceDescriptors.length = imageCount;
    checkCudaErrors(cudaMalloc(&deviceDescriptors.content, bufferSize));
    checkCudaErrors(cudaMemcpy(deviceDescriptors.content, hostDescriptors.content, bufferSize, cudaMemcpyHostToDevice));
    return deviceDescriptors;
}

SpinImage::array<radialIntersectionCountImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<radialIntersectionCountImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<radialIntersectionCountImagePixelType>(hostDescriptors, imageCount);
}

SpinImage::array<spinImagePixelType>
SpinImage::copy::hostDescriptorsToDevice(array<spinImagePixelType> hostDescriptors, size_t imageCount) {
    return copyDescriptorsToDevice<spinImagePixelType>(hostDescriptors, imageCount);
}
