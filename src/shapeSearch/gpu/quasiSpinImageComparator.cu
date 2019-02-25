#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <cuda_runtime_api.h>
#include <shapeSearch/libraryBuildSettings.h>
#include <curand_mtgp32_kernel.h>
#include <tgmath.h>
#include <assert.h>
#include "../../../lib/nvidia-samples-common/nvidia/helper_cuda.h"

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

__device__ void compareSingleImagePair(const array<float> &correlations, size_t imageCount, unsigned int spinImageIndex,
									   const int spinImageElementCount, float sumX, float expectedPart1,
									   unsigned int otherImageIndex, size_t correlationIndex) {
	float threadPartialSumY = 0;
	float threadPartialSquaredSumY = 0;
	float threadPartialMultiplicativeSum = 0;

	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		#pragma unroll
		for (int x = threadIdx.x; x < spinImageWidthPixels; x += blockDim.x)
		{
			float pixelValueX = descriptors.content[spinImageIndex  * spinImageElementCount + (y * spinImageWidthPixels + x)];
			float pixelValueY = descriptors.content[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

			threadPartialSumY += pixelValueY;
			threadPartialSquaredSumY += pixelValueY * pixelValueY;
			threadPartialMultiplicativeSum += pixelValueX * pixelValueY;
		}
	}

#if COMPARISON_BLOCK_SIZE > 32
	__syncthreads();
#endif

	float sumY = warpAllReduceSum(threadPartialSumY);
	float squaredSumY = warpAllReduceSum(threadPartialSquaredSumY);
	float multiplicativeSum = warpAllReduceSum(threadPartialMultiplicativeSum);
	float expectedPart2 = sqrt((float(spinImageElementCount) * squaredSumY) - (sumY * sumY));

#if COMPARISON_BLOCK_SIZE > 32
	__syncthreads();
#endif

	// Ensure we only spam one write
	if(findLowestActiveThreadIndex() == threadIdx.x) {
		float correlation = -1;

		// Avoid zero divisions
		if(expectedPart1 != 0 && expectedPart2 != 0)
		{
			correlation = ((float(spinImageElementCount) * multiplicativeSum) - (sumX * sumY)) / (expectedPart1 * expectedPart2);
		}

		assert(correlationIndex < correlations.length);

		correlations.content[correlationIndex] = correlation;
	}
}

__device__ void computeInitialSums(unsigned int spinImageIndex, float &sumX, float &expectedPart1) {
	float threadPartialSumX = 0;
	float threadPartialSquaredSumX = 0;

	const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

	for (int y = 0; y < spinImageWidthPixels; y++)
	{
#pragma unroll
		for (int x = threadIdx.x; x < spinImageWidthPixels; x += blockDim.x)
		{
			float pixelValue = descriptors.content[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
			threadPartialSumX += pixelValue;
			threadPartialSquaredSumX += pixelValue * pixelValue;
		}
	}

#if COMPARISON_BLOCK_SIZE > 32
	// Needs more work than this, but is fine for single warp
	__syncthreads();
#endif

	sumX = warpAllReduceSum(threadPartialSumX);
	float squaredSumX = warpAllReduceSum(threadPartialSquaredSumX);

	expectedPart1 = float(sqrt((float(spinImageElementCount) * squaredSumX) - (sumX * sumX)));
}

__global__ void doDescriptorComparisonElementWise(DeviceMesh mesh, array<float> correlations, size_t imageCount)
{
	unsigned int spinImageIndex = blockIdx.x;

	// Step 1: Calculate average of spin image
	float sumX;
	float expectedPart1;
	computeInitialSums(spinImageIndex, sumX, expectedPart1);

	unsigned int otherImageIndex = spinImageIndex;
	size_t correlationIndex = spinImageIndex;
	compareSingleImagePair(correlations, imageCount, spinImageIndex, sumX, expectedPart1, otherImageIndex, correlationIndex);
}


__global__ void doDescriptorComparisonComplete(DeviceMesh mesh, array<float> correlations, size_t imageCount)
{
	unsigned int spinImageIndex = blockIdx.x;

	float sumX;
	float expectedPart1;
	computeInitialSums(spinImageIndex, sumX, expectedPart1);

	for(unsigned int otherImageIndex = spinImageIndex + 1; otherImageIndex < descriptors.length; otherImageIndex++) {
		size_t correlationStartIndex = correlations.length - sumCount(imageCount - 1 - spinImageIndex);
		size_t correlationIndex = correlationStartIndex + (otherImageIndex - (spinImageIndex + 1));
		compareSingleImagePair(correlations, imageCount, spinImageIndex, sumX, expectedPart1, otherImageIndex, correlationIndex);

	}
}

array<float> doDescriptorComparison(const DeviceMesh &device_mesh, const array<unsigned int> &device_descriptors,
                                    size_t correlationBufferCount, bool computeComplete) {
	size_t imageCount = device_mesh.vertexCount;

	array<float> device_correlations;
	size_t correlationBufferSize = sizeof(float) * correlationBufferCount;

	checkCudaErrors(cudaMalloc(&device_correlations.content, correlationBufferSize));
	device_correlations.length = correlationBufferCount;

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	cudaMemset(device_correlations.content, 0x0, correlationBufferSize);

	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	if(computeComplete) {
		doDescriptorComparisonComplete<<<imageCount, COMPARISON_BLOCK_SIZE>>> (device_mesh, device_descriptors.content, device_correlations, imageCount);
	} else {
		doDescriptorComparisonElementWise<<<imageCount, COMPARISON_BLOCK_SIZE>>> (device_mesh, device_descriptors.content, device_correlations, imageCount);
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

array<float> compareDescriptorsComplete(DeviceMesh device_mesh, array<unsigned int> device_descriptors)
{
	size_t correlationBufferCount = sumCount(device_mesh.vertexCount - 1);
	return doDescriptorComparison(device_mesh, device_descriptors, correlationBufferCount, true);

}

array<float> compareDescriptorsElementWise(DeviceMesh device_mesh, array<unsigned int> device_descriptors)
{
	size_t correlationBufferCount = device_mesh.vertexCount;
	return doDescriptorComparison(device_mesh, device_descriptors, correlationBufferCount, false);
}