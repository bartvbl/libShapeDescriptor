#include <shapeSearch/gpu/types/DeviceMesh.h>
#include <cuda_runtime.h>
#include <shapeSearch/libraryBuildSettings.h>
#include <curand_mtgp32_kernel.h>
#include <tgmath.h>
#include <assert.h>
#include <iostream>
#include <climits>
#include <cfloat>
#include <chrono>
#include "nvidia/helper_cuda.h"
#include "quasiSpinImageSearcher.cuh"



__inline__ __device__ float warpAllReduceSum(float val) {
	for (int mask = warpSize/2; mask > 0; mask /= 2)
		val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
	return val;
}

template<typename pixelType>
__device__ float computeImagePairCorrelation(pixelType* descriptors,
											 pixelType* otherDescriptors,
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
            const size_t spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

			pixelType pixelValueX = descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
			pixelType pixelValueY = otherDescriptors[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

			float deltaX = float(pixelValueX) - averageX;
			float deltaY = float(pixelValueY) - averageY;

			threadSquaredSumX += deltaX * deltaX;
			threadSquaredSumY += deltaY * deltaY;
			threadMultiplicativeSum += deltaX * deltaY;
		}
	}

	float squaredSumX = sqrt(warpAllReduceSum(threadSquaredSumX));
    float squaredSumY = sqrt(warpAllReduceSum(threadSquaredSumY));
    float multiplicativeSum = warpAllReduceSum(threadMultiplicativeSum);

    float correlation = -1;

    // Avoid zero divisions
    if(squaredSumX != 0 && squaredSumY != 0)
    {
        correlation = multiplicativeSum / (squaredSumX * squaredSumY);
    }

    return correlation;
}

template<typename pixelType>
__device__ float computeImageAverage(pixelType* descriptors, size_t spinImageIndex)
{
	const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

	float threadPartialSum = 0;

	for (int y = 0; y < spinImageWidthPixels; y++)
	{
		for (int x = threadIdx.x; x < spinImageWidthPixels; x += blockDim.x)
		{
			float pixelValue = float(descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)]);
			threadPartialSum += pixelValue;
		}
	}

	return warpAllReduceSum(threadPartialSum) / float(spinImageElementCount);
}

template<typename pixelType>
__global__ void calculateImageAverages(pixelType* images, float* averages) {
	// This kernel assumes one warp per image
	assert(blockDim.x == 32);

	size_t imageIndex = blockIdx.x;

	float average = computeImageAverage<pixelType>(images, imageIndex);

	if(threadIdx.x == 0) {
		averages[imageIndex] = average;
	}
}

template<typename pixelType>
__global__ void generateSearchResults(pixelType* needleDescriptors,
									  pixelType* haystackDescriptors,
									  size_t haystackImageCount,
									  ImageSearchResults* searchResults,
									  float* needleImageAverages,
									  float* haystackImageAverages) {

	size_t needleImageIndex = blockIdx.x;

	// Pearson correlation, which is used as distance measure, means closer to 1 is better
	// We thus initialise the score to the absolute minimum, so that any score is higher.
	static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
	size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
	float threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX}; // FLT_MIN represents smallest POSITIVE float

	float needleImageAverage = needleImageAverages[needleImageIndex];

	for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
		float haystackImageAverage = haystackImageAverages[haystackImageIndex];

		float correlation = computeImagePairCorrelation(needleDescriptors,
														haystackDescriptors,
														needleImageIndex,
														haystackImageIndex,
														needleImageAverage,
														haystackImageAverage);

		// Since most images will not make it into the top ranking, we do a quick check to avoid a search
		// This saves a few instructions.
		if(correlation > __shfl_sync(0xFFFFFFFF, threadSearchResultScores[(SEARCH_RESULT_COUNT / 32) - 1], 31)) {

		    // Issue: does not insert correctly in an empty list
		    /*unsigned int leftBound = 0;
			unsigned int rightBound = blockDim.x - 1;
			unsigned int pivotIndex = (leftBound + rightBound) / 2;

			while(leftBound <= rightBound) {
				pivotIndex = (leftBound + rightBound) / 2;
				float pivotThreadValue = __shfl_sync(0xFFFFFFFF, threadSearchResultScore, pivotIndex);
				if(pivotThreadValue < correlation) {
					leftBound = pivotIndex + 1;
				} else if(pivotThreadValue > correlation) {
					rightBound = pivotIndex - 1;
				} else {
					break;
				}
			}*/

            unsigned int foundIndex = 0;
            for(; foundIndex < SEARCH_RESULT_COUNT; foundIndex++) {
                float threadValue = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[foundIndex / 32], foundIndex % 32);
                if(threadValue < correlation) {
                    break;
                }
            }

            int foundThreadIndex = foundIndex % 32;
            int startBlock = foundIndex / 32;
            const int endBlock = (SEARCH_RESULT_COUNT / 32) - 1;
            for(int block = endBlock; block > startBlock; block--) {
				int targetThread = int(threadIdx.x) - 1;
				int targetBlock = block;
				if(targetThread == -1) {
					targetThread = 31;
					targetBlock = block - 1;
				}

				threadSearchResultScores[targetBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[targetBlock], targetThread);
				threadSearchResultImageIndexes[targetBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[targetBlock], targetThread);
            }
			if(threadIdx.x >= foundIndex) {
				int targetThread = int(threadIdx.x) - 1;

				// Shift all values one thread to the right
				threadSearchResultScores[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[startBlock], targetThread);
				threadSearchResultImageIndexes[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[startBlock], targetThread);

				if(threadIdx.x == foundIndex) {
					threadSearchResultScores[startBlock] = correlation;
					threadSearchResultImageIndexes[startBlock] = haystackImageIndex;
				}
			}

		}
	}

	// Storing search results
	for(int block = 0; block < SEARCH_RESULT_COUNT / 32; block++) {
        searchResults[needleImageIndex].resultIndices[block * SEARCH_RESULT_COUNT + threadIdx.x] = threadSearchResultImageIndexes[block];
        searchResults[needleImageIndex].resultScores[block * SEARCH_RESULT_COUNT + threadIdx.x] = threadSearchResultScores[block];
    }

}

template<typename pixelType>
array<ImageSearchResults> doFindDescriptorsInHaystack(
                                 array<pixelType> device_needleDescriptors,
                                 size_t needleImageCount,
                                 array<pixelType> device_haystackDescriptors,
                                 size_t haystackImageCount)
{
    // Step 1: Compute image averages, since they're constant and are needed for each comparison

	float* device_needleImageAverages;
	float* device_haystackImageAverages;
	checkCudaErrors(cudaMalloc(&device_needleImageAverages, needleImageCount * sizeof(float)));
	checkCudaErrors(cudaMalloc(&device_haystackImageAverages, haystackImageCount * sizeof(float)));

	std::cout << "\t\tComputing image averages.." << std::endl;
	calculateImageAverages<pixelType><<<needleImageCount, 32>>>(device_needleDescriptors.content, device_needleImageAverages);
	calculateImageAverages<pixelType><<<haystackImageCount, 32>>>(device_haystackDescriptors.content, device_haystackImageAverages);
	checkCudaErrors(cudaDeviceSynchronize());

	// Step 2: Perform search

	size_t searchResultBufferSize = needleImageCount * sizeof(ImageSearchResults);
	ImageSearchResults* device_searchResults;
	checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

	std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<needleImageCount, 32>>>(device_needleDescriptors.content,
													device_haystackDescriptors.content,
													haystackImageCount,
													device_searchResults,
													device_needleImageAverages,
													device_haystackImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

	array<ImageSearchResults> searchResults;
	searchResults.content = new ImageSearchResults[needleImageCount];
	searchResults.length = needleImageCount;

	checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

	// Cleanup

	cudaFree(device_needleImageAverages);
	cudaFree(device_haystackImageAverages);
	cudaFree(device_searchResults);

	return searchResults;
}

array<ImageSearchResults> findDescriptorsInHaystack(
		array<classicSpinImagePixelType > device_needleDescriptors,
		size_t needleImageCount,
		array<classicSpinImagePixelType > device_haystackDescriptors,
		size_t haystackImageCount) {
	return doFindDescriptorsInHaystack<classicSpinImagePixelType>(device_needleDescriptors, needleImageCount, device_haystackDescriptors, haystackImageCount);
}

array<ImageSearchResults> findDescriptorsInHaystack(
		array<newSpinImagePixelType> device_needleDescriptors,
		size_t needleImageCount,
		array<newSpinImagePixelType > device_haystackDescriptors,
		size_t haystackImageCount) {
	return doFindDescriptorsInHaystack<newSpinImagePixelType>(device_needleDescriptors, needleImageCount, device_haystackDescriptors, haystackImageCount);
}