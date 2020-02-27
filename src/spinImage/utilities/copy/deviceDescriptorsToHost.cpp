#include <cuda_runtime_api.h>
#include <nvidia/helper_cuda.h>
#include <spinImage/gpu/types/QUICCImages.h>
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

SpinImage::cpu::QUICCIImages SpinImage::copy::QUICCIDescriptorsToHost(SpinImage::gpu::QUICCIImages device_descriptors) {
    size_t descriptorBufferLength = device_descriptors.imageCount;
    size_t descriptorBufferSize = descriptorBufferLength * sizeof(QuiccImage);

    SpinImage::cpu::QUICCIImages host_descriptors;
    host_descriptors.horizontallyIncreasingImages = new QuiccImage[descriptorBufferLength];
    host_descriptors.horizontallyDecreasingImages = new QuiccImage[descriptorBufferLength];
    host_descriptors.imageCount = device_descriptors.imageCount;

    checkCudaErrors(cudaMemcpy(
            host_descriptors.horizontallyIncreasingImages,
            device_descriptors.horizontallyIncreasingImages,
            descriptorBufferSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(
            host_descriptors.horizontallyDecreasingImages,
            device_descriptors.horizontallyDecreasingImages,
            descriptorBufferSize, cudaMemcpyDeviceToHost));

    return host_descriptors;
}