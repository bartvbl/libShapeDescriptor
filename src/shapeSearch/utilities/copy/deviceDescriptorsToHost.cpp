#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include "deviceDescriptorsToHost.h"

array<newSpinImagePixelType> SpinImage::copy::QSIDescriptorsToHost(array<newSpinImagePixelType> device_descriptors, size_t imageCount) {

    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(newSpinImagePixelType) * descriptorBufferLength;

    array<newSpinImagePixelType> host_descriptors;
    host_descriptors.content = new newSpinImagePixelType[imageCount * spinImageWidthPixels * spinImageWidthPixels];
    host_descriptors.length = imageCount;

    checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

    return host_descriptors;
}

array<classicSpinImagePixelType> SpinImage::copy::spinImageDescriptorsToHost(array<classicSpinImagePixelType> device_descriptors, size_t imageCount) {
    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;

    array<classicSpinImagePixelType> host_descriptors;
    host_descriptors.content = new classicSpinImagePixelType[descriptorBufferLength];
    host_descriptors.length = device_descriptors.length;

    checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

    return host_descriptors;
}