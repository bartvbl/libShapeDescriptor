#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include "deviceDescriptorsToHost.h"

SpinImage::array<radialIntersectionCountImagePixelType> SpinImage::copy::RICIDescriptorsToHost(array<radialIntersectionCountImagePixelType> device_descriptors, size_t imageCount) {

    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(radialIntersectionCountImagePixelType) * descriptorBufferLength;

    array<radialIntersectionCountImagePixelType> host_descriptors;
    host_descriptors.content = new radialIntersectionCountImagePixelType[imageCount * spinImageWidthPixels * spinImageWidthPixels];
    host_descriptors.length = imageCount;

    checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

    return host_descriptors;
}

SpinImage::array<spinImagePixelType> SpinImage::copy::spinImageDescriptorsToHost(array<spinImagePixelType> device_descriptors, size_t imageCount) {
    size_t descriptorBufferLength = imageCount * spinImageWidthPixels * spinImageWidthPixels;
    size_t descriptorBufferSize = sizeof(float) * descriptorBufferLength;

    array<spinImagePixelType> host_descriptors;
    host_descriptors.content = new spinImagePixelType[descriptorBufferLength];
    host_descriptors.length = device_descriptors.length;

    checkCudaErrors(cudaMemcpy(host_descriptors.content, device_descriptors.content, descriptorBufferSize, cudaMemcpyDeviceToHost));

    return host_descriptors;
}