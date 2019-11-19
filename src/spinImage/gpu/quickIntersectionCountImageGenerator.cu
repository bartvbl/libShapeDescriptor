#include "quickIntersectionCountImageGenerator.cuh"
#include <nvidia/helper_cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <bitset>

const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
const int unsignedIntegersPerImage = spinImageElementCount / 32;

__global__ void generateQUICCImagesGPU(
        unsigned int* horizontallyIncreasingImages,
        unsigned int* horizontallyDecreasingImages,
        size_t imageCount,
        radialIntersectionCountImagePixelType* RICIDescriptors) {

    const int laneIndex = threadIdx.x % 32;
    const unsigned int imageIndex = blockIdx.x;
    static_assert(spinImageWidthPixels % 32 == 0);

    for(int row = 0; row < spinImageWidthPixels; row++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;

        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    RICIDescriptors[imageIndex * spinImageElementCount + row * spinImageWidthPixels + pixel];

            int targetThread;
            if (laneIndex > 0) {
                targetThread = laneIndex - 1;
            }
            else if (pixel > 0) {
                targetThread = 31;
            } else {
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;

            if (laneIndex == 31) {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
            } else {
                threadNeedleValue = currentNeedlePixelValue;
            }

            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);

            int imageDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

            bool isDeltaIncreasing = imageDelta > 0;
            bool isDeltaDecreasing = imageDelta < 0;

            unsigned int increasingCompressed = __brev(__ballot_sync(0xFFFFFFFF, isDeltaIncreasing));
            unsigned int decreasingCompressed = __brev(__ballot_sync(0xFFFFFFFF, isDeltaDecreasing));

            if(laneIndex == 0) {
                size_t chunkIndex = (imageIndex * unsignedIntegersPerImage) + (row * (spinImageWidthPixels / 32)) + (pixel / 32);
                horizontallyIncreasingImages[chunkIndex] = increasingCompressed;
                horizontallyDecreasingImages[chunkIndex] = decreasingCompressed;
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        }

    }
}

SpinImage::gpu::QUICCIImages SpinImage::gpu::generateQUICCImages(
        array<radialIntersectionCountImagePixelType> RICIDescriptors,
        SpinImage::debug::QUICCIRunInfo* runinfo) {

    // Code is made for unsigned integers. Shorts would require additional logic.
    static_assert(sizeof(radialIntersectionCountImagePixelType) == 4);

    const unsigned int imageCount = RICIDescriptors.length;
    size_t imageSequenceSize = imageCount * unsignedIntegersPerImage * sizeof(unsigned int);

    unsigned int* device_horizontallyIncreasingImages;
    unsigned int* device_horizontallyDecreasingImages;

    checkCudaErrors(cudaMalloc(&device_horizontallyIncreasingImages, imageSequenceSize));
    checkCudaErrors(cudaMalloc(&device_horizontallyDecreasingImages, imageSequenceSize));

    dim3 gridDimensions = {imageCount, 1, 1};
    dim3 blockDimensions = {32, 1, 1};

    generateQUICCImagesGPU<<<gridDimensions, blockDimensions>>>(
        device_horizontallyIncreasingImages,
        device_horizontallyDecreasingImages,
        imageCount,
        RICIDescriptors.content);
    checkCudaErrors(cudaDeviceSynchronize());

    SpinImage::gpu::QUICCIImages descriptors;
    descriptors.horizontallyIncreasingImages = device_horizontallyIncreasingImages;
    descriptors.horizontallyDecreasingImages = device_horizontallyDecreasingImages;
    descriptors.imageCount = imageCount;

    return descriptors;
}