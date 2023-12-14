#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif

#include <shapeDescriptor/shapeDescriptor.h>
#include <cassert>
#include <iostream>
#include <climits>
#include <cfloat>
#include <chrono>
#include <typeinfo>

const unsigned int warpCount = 16;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__inline__ __device__ float warpAllReduceSum(float val) {
	for (int mask = warpSize/2; mask > 0; mask /= 2)
		val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
	return val;
}

__device__ float computeSpinImagePairCorrelationGPU(
        ShapeDescriptor::SpinImageDescriptor* descriptors,
        ShapeDescriptor::SpinImageDescriptor* otherDescriptors,
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
			pixelValueX = descriptors[spinImageIndex].contents[y * spinImageWidthPixels + x];
			pixelValueY = otherDescriptors[otherImageIndex].contents[y * spinImageWidthPixels + x];

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

__global__ void calculateImageAverages(ShapeDescriptor::SpinImageDescriptor* images, float* averages) {
	size_t imageIndex = blockIdx.x;

	// Only support up to 32 warps
	__shared__ float warpSums[32];

	if(threadIdx.x < 32) {
	    warpSums[threadIdx.x] = 0;
	}

	__syncthreads();

    const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

    float threadPartialSum = 0;

    for (int x = threadIdx.x; x < spinImageElementCount; x += blockDim.x)
    {
        float pixelValue = float(images[imageIndex].contents[x]);
        threadPartialSum += pixelValue;
    }

    float warpSum = warpAllReduceSum(threadPartialSum);

    if(threadIdx.x % 32 == 0) {
        warpSums[threadIdx.x / 32] = warpSum;
    }

    __syncthreads();

    if(threadIdx.x < 32) {
        float threadSum = warpSums[threadIdx.x];
        threadSum = warpAllReduceSum(threadSum);
        if(threadIdx.x == 0) {
            averages[imageIndex] = threadSum / float(spinImageElementCount);
        }
    }
}

__global__ void generateSearchResults(ShapeDescriptor::SpinImageDescriptor* needleDescriptors,
									  size_t needleImageCount,
                                      ShapeDescriptor::SpinImageDescriptor* haystackDescriptors,
									  size_t haystackImageCount,
                                      ShapeDescriptor::SearchResults<float>* searchResults,
									  float* needleImageAverages,
									  float* haystackImageAverages) {

	size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

	if(needleImageIndex >= needleImageCount) {
	    return;
	}

	// Pearson correlation, which is used as distance measure, means closer to 1 is better
	// We thus initialise the score to the absolute minimum, so that any score is higher.
	static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
	size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
	float threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX}; // FLT_MIN represents smallest POSITIVE float

    const int blockCount = (SEARCH_RESULT_COUNT / 32);

	float needleImageAverage = needleImageAverages[needleImageIndex];

	for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
		float haystackImageAverage = haystackImageAverages[haystackImageIndex];

		float correlation = computeSpinImagePairCorrelationGPU(
		        needleDescriptors,
		        haystackDescriptors,
		        needleImageIndex,
		        haystackImageIndex,
		        needleImageAverage,
		        haystackImageAverage);

		// Since most images will not make it into the top ranking, we do a quick check to avoid a search
		// This saves a few instructions.
		if(correlation > __shfl_sync(0xFFFFFFFF, threadSearchResultScores[(SEARCH_RESULT_COUNT / 32) - 1], 31)) {
		    unsigned int foundIndex = 0;
            for(int block = 0; block < blockCount; block++) {
                bool threadExceeds = threadSearchResultScores[block] < correlation;
                unsigned int bitString = __ballot_sync(0xFFFFFFFF, threadExceeds);
                unsigned int firstSet = __ffs(bitString) - 1;

                if(firstSet < 32) {
                    foundIndex = (block * 32) + (firstSet);
                    break;
                }
            }

            int startBlock = foundIndex / 32;
            const int endBlock = blockCount - 1;
			const int laneID = threadIdx.x % 32;

            // We first shift all values to the right for "full" 32-value blocks
            // Afterwards, we do one final iteration to shift only the values that are
            // block will never be 0, which ensures the loop body does not go out of range
            for(int block = endBlock; block > startBlock; block--) {
                int sourceThread = laneID - 1;
                int sourceBlock = block;

                if(laneID == 0) {
                    sourceThread = 31;
                }
                if(laneID == 31) {
                    sourceBlock = block - 1;
                }

				threadSearchResultScores[block] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[sourceBlock], sourceThread);
				threadSearchResultImageIndexes[block] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[sourceBlock], sourceThread);
            }

            // This shifts over values in the block where we're inserting the new value.
            // As such it requires some more fine-grained control.
			if(laneID >= foundIndex % 32) {
				int targetThread = laneID - 1;

				threadSearchResultScores[startBlock] = __shfl_sync(__activemask(), threadSearchResultScores[startBlock], targetThread);
				threadSearchResultImageIndexes[startBlock] = __shfl_sync(__activemask(), threadSearchResultImageIndexes[startBlock], targetThread);

				if(laneID == foundIndex % 32) {
					threadSearchResultScores[startBlock] = correlation;
					threadSearchResultImageIndexes[startBlock] = haystackImageIndex;
				}
			}

		}
	}

    const unsigned int laneID = threadIdx.x % 32;
	// Storing search results
	for(int block = 0; block < blockCount; block++) {
        searchResults[needleImageIndex].indices[block * 32 + laneID] = threadSearchResultImageIndexes[block];
        searchResults[needleImageIndex].scores[block * 32 + laneID] = threadSearchResultScores[block];
    }

}

#endif

ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<float>> ShapeDescriptor::findSpinImagesInHaystack(
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackDescriptors) {

    // Step 1: Compute image averages, since they're constant and are needed for each comparison
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
	float* device_needleImageAverages;
	float* device_haystackImageAverages;
	checkCudaErrors(cudaMalloc(&device_needleImageAverages, device_needleDescriptors.length * sizeof(float)));
	checkCudaErrors(cudaMalloc(&device_haystackImageAverages, device_haystackDescriptors.length * sizeof(float)));

	std::cout << "\t\tComputing image averages.." << std::endl;
	calculateImageAverages<<<device_needleDescriptors.length, 32>>>(device_needleDescriptors.content, device_needleImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());
	calculateImageAverages<<<device_haystackDescriptors.length, 32>>>(device_haystackDescriptors.content, device_haystackImageAverages);
	checkCudaErrors(cudaDeviceSynchronize());

	// Step 2: Perform search

	size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(ShapeDescriptor::SearchResults<float>);
    ShapeDescriptor::SearchResults<float>* device_searchResults;
	checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

	std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(device_needleDescriptors.length / warpCount) + 1, 32 * warpCount>>>(
                                                    device_needleDescriptors.content,
													device_needleDescriptors.length,
													device_haystackDescriptors.content,
													device_haystackDescriptors.length,
													device_searchResults,
													device_needleImageAverages,
													device_haystackImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

	ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<float>> searchResults;
	searchResults.content = new ShapeDescriptor::SearchResults<float>[device_needleDescriptors.length];
	searchResults.length = device_needleDescriptors.length;

	checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

	// Cleanup

	cudaFree(device_needleImageAverages);
	cudaFree(device_haystackImageAverages);
	cudaFree(device_searchResults);

	return searchResults;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}




















const int indexBasedWarpCount = 16;
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeSpinImageSearchResultIndices(
        ShapeDescriptor::SpinImageDescriptor* needleDescriptors,
        ShapeDescriptor::SpinImageDescriptor* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults,
        float* needleImageAverages,
        float* haystackImageAverages) {

    size_t needleImageIndex = blockIdx.x;

	__shared__ ShapeDescriptor::SpinImageDescriptor referenceImage;
	for(unsigned int index = threadIdx.x; index < spinImageWidthPixels * spinImageWidthPixels; index += blockDim.x) {
		referenceImage.contents[index] = needleDescriptors[needleImageIndex].contents[index];
	}

	__syncthreads();

	float needleImageAverage = needleImageAverages[needleImageIndex];
	float correspondingImageAverage = haystackImageAverages[needleImageIndex];

	float referenceCorrelation = computeSpinImagePairCorrelationGPU(
	        &referenceImage,
	        haystackDescriptors,
	        0,
	        needleImageIndex,
	        needleImageAverage,
	        correspondingImageAverage);

	// No image pair can have a better correlation than 1, so we can just stop the search right here
	if(referenceCorrelation == 1) {
		return;
	}

	unsigned int searchResultRank = 0;

	for(size_t haystackImageIndex = threadIdx.x / 32; haystackImageIndex < haystackImageCount; haystackImageIndex += indexBasedWarpCount) {
		if(needleImageIndex == haystackImageIndex) {
			continue;
		}

		float haystackImageAverage = haystackImageAverages[haystackImageIndex];

		float correlation = computeSpinImagePairCorrelationGPU(
		        &referenceImage,
		        haystackDescriptors,
		        0,
		        haystackImageIndex,
		        needleImageAverage,
		        haystackImageAverage);

		// We've found a result that's better than the reference one. That means this search result would end up
		// above ours in the search result list. We therefore move our search result down by 1.
		if(correlation > referenceCorrelation) {
			searchResultRank++;
		}
	}

	// Since we're running multiple warps, we need to add all indices together to get the correct ranks
	if(threadIdx.x % 32 == 0) {
		atomicAdd(&searchResults[needleImageIndex], searchResultRank);
	}
}
#endif

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::computeSpinImageSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackDescriptors,
        ShapeDescriptor::SISearchExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
	auto executionStart = std::chrono::steady_clock::now();

	size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
	unsigned int* device_searchResults;
	checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
	checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    float* device_needleImageAverages;
    float* device_haystackImageAverages;
    checkCudaErrors(cudaMalloc(&device_needleImageAverages, device_needleDescriptors.length * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_haystackImageAverages, device_haystackDescriptors.length * sizeof(float)));

    auto averagingStart = std::chrono::steady_clock::now();

        calculateImageAverages<<<device_needleDescriptors.length, 32>>>(device_needleDescriptors.content, device_needleImageAverages);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        calculateImageAverages<<<device_haystackDescriptors.length, 32>>>(device_haystackDescriptors.content, device_haystackImageAverages);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds averagingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - averagingStart);

	auto searchStart = std::chrono::steady_clock::now();

    computeSpinImageSearchResultIndices<<<device_needleDescriptors.length, 32 * indexBasedWarpCount>>>(
                    device_needleDescriptors.content,
                    device_haystackDescriptors.content,
                    device_haystackDescriptors.length,
                    device_searchResults,
                    device_needleImageAverages,
                    device_haystackImageAverages);

	checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

	std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

	ShapeDescriptor::cpu::array<unsigned int> resultIndices;
	resultIndices.content = new unsigned int[device_needleDescriptors.length];
	resultIndices.length = device_needleDescriptors.length;

	checkCudaErrors(cudaMemcpy(resultIndices.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

	// Cleanup

    cudaFree(device_needleImageAverages);
    cudaFree(device_haystackImageAverages);
	cudaFree(device_searchResults);

	std::chrono::milliseconds executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - executionStart);

	if(executionTimes != nullptr) {
        executionTimes->searchExecutionTimeSeconds = double(searchDuration.count()) / 1000.0;
        executionTimes->totalExecutionTimeSeconds = double(executionDuration.count()) / 1000.0;
        executionTimes->averagingExecutionTimeSeconds = double(averagingDuration.count()) / 1000.0;
	}

	return resultIndices;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}




#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeElementWiseSIEuclideanDistances(
        ShapeDescriptor::SpinImageDescriptor* descriptors,
        ShapeDescriptor::SpinImageDescriptor* correspondingDescriptors,
        float* distances) {
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    size_t needleImageIndex = blockIdx.x;

    float threadSquaredSum = 0;

    for(unsigned int i = threadIdx.x; i < spinImageWidthPixels * spinImageWidthPixels; i += blockDim.x) {
        spinImagePixelType descriptorPixelValue = descriptors[needleImageIndex].contents[i];
        spinImagePixelType correspondingPixelValue = correspondingDescriptors[needleImageIndex].contents[i];
        spinImagePixelType pixelDelta = descriptorPixelValue - correspondingPixelValue;
        threadSquaredSum += pixelDelta * pixelDelta;
    }

    float totalSquaredSum = warpAllReduceSum(threadSquaredSum);

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = sqrt(totalSquaredSum);
    }
}
#endif

ShapeDescriptor::cpu::array<float> ShapeDescriptor::computeSIElementWiseEuclideanDistances(
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_descriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_correspondingDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<float> device_distances(device_descriptors.length);

    computeElementWiseSIEuclideanDistances<<<device_descriptors.length, 32>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            device_distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<float> distances = device_distances.copyToCPU();

    ShapeDescriptor::free(device_distances);

    return distances;
    #else
throw std::runtime_error(cudaMissingErrorMessage);
#endif
}

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeElementWiseSIPearsonCorrelations(
        ShapeDescriptor::SpinImageDescriptor* descriptors,
        ShapeDescriptor::SpinImageDescriptor* correspondingDescriptors,
        const float* descriptorAverages,
        const float* correspondingDescriptorAverages,
        float* distances) {
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    size_t needleImageIndex = blockIdx.x;
    float descriptorAverage = descriptorAverages[needleImageIndex];
    float correspondingAverage = correspondingDescriptorAverages[needleImageIndex];

    float distanceScore = computeSpinImagePairCorrelationGPU(
            descriptors, correspondingDescriptors,
            needleImageIndex, needleImageIndex,
            descriptorAverage, correspondingAverage);

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = distanceScore;
    }
}
#endif

ShapeDescriptor::cpu::array<float> ShapeDescriptor::computeSIElementWisePearsonCorrelations(
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_descriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_correspondingDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<float> descriptorAverages(device_descriptors.length);
    ShapeDescriptor::gpu::array<float> correspondingAverages(device_correspondingDescriptors.length);

    calculateImageAverages<<<device_descriptors.length, 32>>>(device_descriptors.content, descriptorAverages.content);
    calculateImageAverages<<<device_descriptors.length, 32>>>(device_correspondingDescriptors.content, correspondingAverages.content);
    checkCudaErrors(cudaDeviceSynchronize());

    ShapeDescriptor::gpu::array<float> device_distances(device_descriptors.length);

    computeElementWiseSIPearsonCorrelations<<<device_descriptors.length, 32>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            descriptorAverages.content, correspondingAverages.content, device_distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<float> distances = device_distances.copyToCPU();

    ShapeDescriptor::free(device_distances);
    ShapeDescriptor::free(descriptorAverages);
    ShapeDescriptor::free(correspondingAverages);

    return distances;
#else
throw std::runtime_error(cudaMissingErrorMessage);
#endif
}