#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <cuda_runtime_api.h>
#include <shapeSearch/libraryBuildSettings.h>
#include <curand_mtgp32_kernel.h>
#include <tgmath.h>
#include <assert.h>
#include "nvidia/helper_cuda.h"

#define COMPARISON_BLOCK_SIZE 32

__device__ __host__ size_t sumCount(size_t count) {
	return (count * (count + 1)) / 2;
}

__device__ __inline__ unsigned int findLowestActiveThreadIndex()
{
	unsigned int activeThreadMask = __activemask();
	unsigned int correctedActiveThreadMask = __brev(activeThreadMask);
	// Find the lowest bit set to 1.
	// Since __ffs() returns a least significant bit, we need to convert it to a lane index.
	unsigned int lowestActiveIndex = __clz(correctedActiveThreadMask);
	return lowestActiveIndex;
}

__inline__ __device__ float warpAllReduceSum(float val) {
	for (int mask = warpSize/2; mask > 0; mask /= 2)
		val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
	return val;
}

__device__ float computeImagePairCorrelation(array<newSpinImagePixelType> descriptors,
                                             array<newSpinImagePixelType> otherDescriptors,
                                             size_t spinImageIndex,
                                             size_t otherImageIndex,
                                             float averageX, float averageY) {
	float threadSquaredSumX = 0;
	float threadSquaredSumY = 0;
	float threadMultiplicativeSum = 0;

	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = threadIdx.x; x < spinImageWidthPixels; x += blockDim.x)
		{
            const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

            newSpinImagePixelType pixelValueX = descriptors.content[spinImageIndex  * spinImageElementCount + (y * spinImageWidthPixels + x)];
            newSpinImagePixelType pixelValueY = otherDescriptors.content[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

			float deltaX = pixelValueX - averageX;
			float deltaY = pixelValueY - averageY;

			threadSquaredSumX += deltaX * deltaX;
			threadSquaredSumY += deltaY * deltaY;
			threadMultiplicativeSum += deltaX * deltaY;
		}
	}

	float squaredSumX = warpAllReduceSum(threadSquaredSumX);
    float squaredSumY = warpAllReduceSum(threadSquaredSumY);
    float multiplicativeSum = warpAllReduceSum(threadMultiplicativeSum);

    float correlation = -1;

    // Avoid zero divisions
    if(squaredSumX != 0 && squaredSumY != 0)
    {
        correlation = multiplicativeSum / (squaredSumX * squaredSumY);
    }

    return correlation;
}



__device__ float computeImageAverage(array<newSpinImagePixelType> descriptors, unsigned int spinImageIndex)
{
	const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

	float threadPartialSum = 0;

	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = threadIdx.x; x < spinImageWidthPixels; x += blockDim.x)
		{
			float pixelValue = descriptors.content[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
			threadPartialSum += pixelValue;
		}
	}

	return warpAllReduceSum(threadPartialSum) / float(spinImageElementCount);
}




__device__ void writeCorrelationValue(const array<float> &correlations, float correlation, size_t correlationIndex) {
    // Ensure we only spam one write
    if(findLowestActiveThreadIndex() == threadIdx.x) {
        correlations.content[correlationIndex] = correlation;
    }
}



__global__ void doDescriptorComparisonElementWise(array<newSpinImagePixelType> descriptors,
                                                  array<newSpinImagePixelType> otherDescriptors,
                                                  array<float> correlations,
                                                  size_t imageCount)
{
	const unsigned int spinImageIndex = blockIdx.x;


	float averageX = computeImageAverage(descriptors, spinImageIndex);
	float averageY = computeImageAverage(otherDescriptors, spinImageIndex);

	unsigned int otherImageIndex = spinImageIndex;
	size_t correlationIndex = spinImageIndex;

	float correlation = computeImagePairCorrelation(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);

    writeCorrelationValue(correlations, correlation, correlationIndex);
}



__global__ void doDescriptorComparisonComplete(array<newSpinImagePixelType> descriptors,
                                               array<newSpinImagePixelType> otherDescriptors,
                                               array<float> correlations,
                                               size_t imageCount)
{
	const unsigned int spinImageIndex = blockIdx.x;

    float averageX = computeImageAverage(descriptors, spinImageIndex);

	for(unsigned int otherImageIndex = spinImageIndex + 1; otherImageIndex < imageCount; otherImageIndex++) {
        float averageY = computeImageAverage(otherDescriptors, otherImageIndex);

		float correlation = computeImagePairCorrelation(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);

		size_t correlationStartIndex = correlations.length - sumCount(imageCount - 1 - spinImageIndex);
		size_t correlationIndex = correlationStartIndex + (otherImageIndex - (spinImageIndex + 1));
        writeCorrelationValue(correlations, correlation, correlationIndex);
	}
}



array<float> doDescriptorComparison(const array<newSpinImagePixelType> &device_descriptors,
                                    const array<newSpinImagePixelType> &device_otherDescriptors,
                                    size_t imageCount,
                                    size_t correlationBufferCount,
                                    bool computeComplete)
{

	array<float> device_correlations;
	size_t correlationBufferSize = sizeof(float) * correlationBufferCount;

	checkCudaErrors(cudaMalloc(&device_correlations.content, correlationBufferSize));
	device_correlations.length = correlationBufferCount;

    checkCudaErrors(cudaMemset(device_correlations.content, 0x0, correlationBufferSize));

#if COMPARISON_BLOCK_SIZE != 32
#error "Kernel code has been written for one warp per block"
#endif

	if(computeComplete) {
		doDescriptorComparisonComplete<<<imageCount, COMPARISON_BLOCK_SIZE>>> (device_descriptors, device_otherDescriptors, device_correlations, imageCount);
	} else {
		doDescriptorComparisonElementWise<<<imageCount, COMPARISON_BLOCK_SIZE>>> (device_descriptors, device_otherDescriptors, device_correlations, imageCount);
	}

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	array<float> host_correlations;
	host_correlations.content = new float[correlationBufferCount];
	host_correlations.length = correlationBufferCount;

	cudaMemcpy(host_correlations.content, device_correlations.content, correlationBufferCount * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(device_correlations.content);

	return host_correlations;
}



array<float> compareDescriptorsComplete(array<newSpinImagePixelType> device_descriptors,
                                        array<newSpinImagePixelType> device_otherDescriptors,
                                        size_t imageCount)
{
	size_t correlationBufferCount = sumCount(imageCount - 1);
	return doDescriptorComparison(device_descriptors, device_otherDescriptors, imageCount, correlationBufferCount, true);

}



array<float> compareDescriptorsElementWise(array<newSpinImagePixelType> device_descriptors,
                                           array<newSpinImagePixelType> device_otherDescriptors,
                                           size_t imageCount)
{
	size_t correlationBufferCount = imageCount;
	return doDescriptorComparison(device_descriptors, device_otherDescriptors, imageCount, correlationBufferCount, false);
}