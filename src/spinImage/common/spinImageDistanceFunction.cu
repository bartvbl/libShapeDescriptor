#include <spinImage/libraryBuildSettings.h>
#include "spinImageDistanceFunction.cuh"

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ float computeSpinImagePairCorrelationGPU(
        spinImagePixelType* descriptors,
        spinImagePixelType* otherDescriptors,
        size_t spinImageIndex,
        size_t otherImageIndex,
        float averageX, float averageY) {

    float threadSquaredSumX = 0;
    float threadSquaredSumY = 0;
    float threadMultiplicativeSum = 0;

    spinImagePixelType pixelValueX;
    spinImagePixelType pixelValueY;

    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        const int warpSize = 32;
        for (int x = threadIdx.x % 32; x < spinImageWidthPixels; x += warpSize)
        {
            const size_t spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

            pixelValueX = descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
            pixelValueY = otherDescriptors[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

            float deltaX = float(pixelValueX) - averageX;
            float deltaY = float(pixelValueY) - averageY;

            threadSquaredSumX += deltaX * deltaX;
            threadSquaredSumY += deltaY * deltaY;
            threadMultiplicativeSum += deltaX * deltaY;
        }
    }

    float squaredSumX = float(sqrt(warpAllReduceSum(threadSquaredSumX)));
    float squaredSumY = float(sqrt(warpAllReduceSum(threadSquaredSumY)));
    float multiplicativeSum = warpAllReduceSum(threadMultiplicativeSum);

    float correlation = multiplicativeSum / (squaredSumX * squaredSumY);

    return correlation;
}


__host__ float computeSpinImagePairCorrelationCPU(
        spinImagePixelType* descriptors,
        spinImagePixelType* otherDescriptors,
        size_t spinImageIndex,
        size_t otherImageIndex,
        float averageX, float averageY) {

    float squaredSumX = 0;
    float squaredSumY = 0;
    float multiplicativeSum = 0;

    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels; x++)
        {
            const size_t spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

            spinImagePixelType pixelValueX = descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
            spinImagePixelType pixelValueY = otherDescriptors[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

            float deltaX = float(pixelValueX) - averageX;
            float deltaY = float(pixelValueY) - averageY;

            squaredSumX += deltaX * deltaX;
            squaredSumY += deltaY * deltaY;
            multiplicativeSum += deltaX * deltaY;
        }
    }

    squaredSumX = std::sqrt(squaredSumX);
    squaredSumY = std::sqrt(squaredSumY);

    // Assuming non-constant images
    // Will return NaN otherwise
    float correlation = multiplicativeSum / (squaredSumX * squaredSumY);

    return correlation;
}