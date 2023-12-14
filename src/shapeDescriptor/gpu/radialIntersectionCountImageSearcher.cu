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

#ifndef warpSize
#define warpSize 32
#endif

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__inline__ __device__ int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

const int indexBasedWarpCount = 16;

__device__ int computeImageSquaredSumGPU(const ShapeDescriptor::RICIDescriptor &needleImage) {

    const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
    const int laneIndex = threadIdx.x % 32;

    unsigned int threadSquaredSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");

    // Scores are computed one row at a time.
    // We differentiate between rows to ensure the final pixel of the previous row does not
    // affect the first pixel of the next one.
    for(int pixel = 0; pixel < spinImageElementCount; pixel++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;
        radialIntersectionCountImagePixelType currentNeedlePixelValue =
                needleImage.contents[pixel];

        int targetThread;
        if (laneIndex > 0) {
            targetThread = laneIndex - 1;
        } else if (pixel > 0) {
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
        int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);

        threadSquaredSum += unsigned(needleDelta * needleDelta);
    }

    int squaredSum = warpAllReduceSum(threadSquaredSum);

    return squaredSum;
}

__device__ size_t compareConstantRadialIntersectionCountImagePairGPU(
        const ShapeDescriptor::RICIDescriptor* needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::RICIDescriptor* haystackImages,
        const size_t haystackImageIndex) {

    const int laneIndex = threadIdx.x % 32;

    // Assumption: there will never be an intersection count over 65535 (which would cause this to overflow)
    size_t threadDeltaSquaredSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");


    // Scores are computed one row at a time.
    // We differentiate between rows to ensure the final pixel of the previous row does not
    // affect the first pixel of the next one.
    for(int row = 0; row < spinImageWidthPixels; row++) {
        // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    needleImages[needleImageIndex].contents[row * spinImageWidthPixels + pixel];
            radialIntersectionCountImagePixelType currentHaystackPixelValue =
                    haystackImages[haystackImageIndex].contents[row * spinImageWidthPixels + pixel];

            // This bit handles the case where an image is completely constant.
            // In that case, we use the absolute sum of squares as a distance function instead
            int imageDelta = int(currentNeedlePixelValue) - int(currentHaystackPixelValue);
            threadDeltaSquaredSum += unsigned(imageDelta * imageDelta); // TODO: size_t?
        }
    }

    // image is constant.
    // In those situations, imageScore would always be 0
    // So we use an unfiltered squared sum instead
    size_t imageScore = warpAllReduceSum(threadDeltaSquaredSum);

    return imageScore;
}







__device__ int compareRadialIntersectionCountImagePairGPU(
        const ShapeDescriptor::RICIDescriptor* needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::RICIDescriptor* haystackImages,
        const size_t haystackImageIndex,
        const int distanceToBeat = INT_MAX) {

    int threadScore = 0;
    const int laneIndex = threadIdx.x % 32;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes an image whose width is a multiple of the warp size");

    // Scores are computed one row at a time.
    // We differentiate between rows to ensure the final pixel of the previous row does not
    // affect the first pixel of the next one.
    for(int row = 0; row < spinImageWidthPixels; row++) {
        radialIntersectionCountImagePixelType previousWarpLastNeedlePixelValue = 0;
        radialIntersectionCountImagePixelType previousWarpLastHaystackPixelValue = 0;
        // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            radialIntersectionCountImagePixelType currentNeedlePixelValue =
                    needleImages[needleImageIndex].contents[row * spinImageWidthPixels + pixel];
            radialIntersectionCountImagePixelType currentHaystackPixelValue =
                    haystackImages[haystackImageIndex].contents[row * spinImageWidthPixels + pixel];

            // To save on memory bandwidth, we use shuffle instructions to pass around other values needed by the
            // distance computation. We first need to use some logic to determine which thread should read from which
            // other thread.
            int targetThread;
            if (laneIndex > 0) {
                // Each thread reads from the previous one
                targetThread = laneIndex - 1;
            }
            // For these last two: lane index is 0. The first pixel of each row receives special treatment, as
            // there is no pixel left of it you can compute a difference over
            else if (pixel > 0) {
                // If pixel is not the first pixel in the row, we read the difference value from the previous iteration
                targetThread = 31;
            } else {
                // If the pixel is the leftmost pixel in the row, we give targetThread a dummy value that will always
                // result in a pixel delta of zero: our own thread with ID 0.
                targetThread = 0;
            }

            radialIntersectionCountImagePixelType threadNeedleValue = 0;
            radialIntersectionCountImagePixelType threadHaystackValue = 0;

            // Here we determine the outgoing value of the shuffle.
            // If we're the last thread in the warp, thread 0 will read from us if we're not processing the first batch
            // of 32 pixels in the row. Since in that case thread 0 will read from itself, we can simplify that check
            // into whether we are lane 31.
            if (laneIndex == 31) {
                threadNeedleValue = previousWarpLastNeedlePixelValue;
                threadHaystackValue = previousWarpLastHaystackPixelValue;
            } else {
                threadNeedleValue = currentNeedlePixelValue;
                threadHaystackValue = currentHaystackPixelValue;
            }

            // Exchange "previous pixel" values through shuffle instructions
            radialIntersectionCountImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);
            radialIntersectionCountImagePixelType previousHaystackPixelValue = __shfl_sync(0xFFFFFFFF, threadHaystackValue,
                                                                                           targetThread);

            // The distance measure this function computes is based on deltas between pairs of pixels
            int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
            int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

            // This if statement makes a massive difference in the clutter resistant performance of this method
            // It only counts least squares differences if the needle image has a change in intersection count
            // Which is usually something very specific to that object.
            if (needleDelta != 0) {
                threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
            previousWarpLastHaystackPixelValue = currentHaystackPixelValue;
        }
#if ENABLE_RICI_COMPARISON_EARLY_EXIT
        // At the end of each block of 8 rows, check whether we can do an early exit
        // This also works for the constant image
        if(row != (spinImageWidthPixels - 1)) {
            int intermediateDistance = warpAllReduceSum(threadScore);
            if(intermediateDistance >= distanceToBeat) {
                return intermediateDistance;
            }
        }
#endif
    }

    int imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}

__global__ void computeRadialIntersectionCountImageSearchResultIndices(
        const ShapeDescriptor::RICIDescriptor* needleDescriptors,
        ShapeDescriptor::RICIDescriptor* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ ShapeDescriptor::RICIDescriptor referenceImage;
    for(unsigned int index = threadIdx.x; index < spinImageWidthPixels * spinImageWidthPixels; index += blockDim.x) {
        referenceImage.contents[index] = needleDescriptors[needleImageIndex].contents[index];
    }

    __syncthreads();

    int referenceScore;

    int needleSquaredSum = computeImageSquaredSumGPU(referenceImage);

    bool needleImageIsConstant = needleSquaredSum == 0;

    if(!needleImageIsConstant) {
        referenceScore = compareRadialIntersectionCountImagePairGPU(
                &referenceImage, 0,
                haystackDescriptors, needleImageIndex);
    } else {
        referenceScore = compareConstantRadialIntersectionCountImagePairGPU(
                &referenceImage, 0,
                haystackDescriptors, needleImageIndex);
    }

    // If the reference distance is 0, no image pair can beat the score. As such we can just skip it.
    if(referenceScore == 0) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackImageIndex = threadIdx.x / 32; haystackImageIndex < haystackImageCount; haystackImageIndex += indexBasedWarpCount) {
        // Don't include the reference image pair in the computation of the reference pair search result index
        if (needleImageIndex == haystackImageIndex) {
            continue;
        }

        // For the current image pair, compute the distance score
        // This depends on whether the needle image is constant
        int pairScore;
        if(!needleImageIsConstant) {
            // If there's variation in the image, we'll use the regular distance function
            pairScore = compareRadialIntersectionCountImagePairGPU(
                    &referenceImage, 0, haystackDescriptors,
                    haystackImageIndex, referenceScore);
        } else {
            // If the image is constant, we use sum of squares as a fallback
            pairScore = compareConstantRadialIntersectionCountImagePairGPU(
                    &referenceImage, 0,
                    haystackDescriptors, haystackImageIndex);
        }

        // We found a better search result that will end up higher in the results list
        // Therefore we move our reference image one result down
        if(pairScore < referenceScore) {
            searchResultRank++;
        }
    }

    // Since we're running multiple warps, we need to add all indices together to get the correct ranks
    // All threads have now computed the same value for searchResultRank, so no need to do a reduction.
    if(threadIdx.x % 32 == 0) {
        atomicAdd(&searchResults[needleImageIndex], searchResultRank);
    }
}
#endif

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::computeRadialIntersectionCountImageSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackDescriptors,
        ShapeDescriptor::RICISearchExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeRadialIntersectionCountImageSearchResultIndices <<< device_needleDescriptors.length, 32 * indexBasedWarpCount >>> (device_needleDescriptors.content,
      device_haystackDescriptors.content,
      device_haystackDescriptors.length,
      device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

    ShapeDescriptor::cpu::array<unsigned int> resultIndices;
    resultIndices.content = new unsigned int[device_needleDescriptors.length];
    resultIndices.length = device_needleDescriptors.length;

    checkCudaErrors(cudaMemcpy(resultIndices.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    std::chrono::milliseconds executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - executionStart);

    if(executionTimes != nullptr) {
        executionTimes->searchExecutionTimeSeconds = double(searchDuration.count()) / 1000.0;
        executionTimes->totalExecutionTimeSeconds = double(executionDuration.count()) / 1000.0;
    }

    return resultIndices;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}































const unsigned int warpCount = 16;
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void generateSearchResults(ShapeDescriptor::RICIDescriptor* needleDescriptors,
                                      size_t needleImageCount,
                                      ShapeDescriptor::RICIDescriptor* haystackDescriptors,
                                      size_t haystackImageCount,
                                      ShapeDescriptor::SearchResults<unsigned int>* searchResults) {

    size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

    if(needleImageIndex >= needleImageCount) {
        return;
    }

    static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
    size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    int threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

    const int blockCount = (SEARCH_RESULT_COUNT / 32);

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
        int score = compareRadialIntersectionCountImagePairGPU(
                needleDescriptors,
                needleImageIndex,
                haystackDescriptors,
                haystackImageIndex);

        // Since most images will not make it into the top ranking, we do a quick check to avoid a search
        // This saves a few instructions.
        if(score < __shfl_sync(0xFFFFFFFF, threadSearchResultScores[(SEARCH_RESULT_COUNT / 32) - 1], 31)) {
            unsigned int foundIndex = 0;
            for(int block = 0; block < blockCount; block++) {
                bool threadExceeds = threadSearchResultScores[block] > score;
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
                    threadSearchResultScores[startBlock] = score;
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

ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<unsigned int>> ShapeDescriptor::findRadialIntersectionCountImagesInHaystack(
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(ShapeDescriptor::SearchResults<unsigned int>);
    ShapeDescriptor::SearchResults<unsigned int>* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

    std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(device_needleDescriptors.length / warpCount) + 1, 32 * warpCount>>>(
            device_needleDescriptors.content,
            device_needleDescriptors.length,
            device_haystackDescriptors.content,
            device_haystackDescriptors.length,
            device_searchResults);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

    ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<unsigned int>> searchResults;
    searchResults.content = new ShapeDescriptor::SearchResults<unsigned int>[device_needleDescriptors.length];
    searchResults.length = device_needleDescriptors.length;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    return searchResults;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}


















#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeElementWiseRICIDistances(
        ShapeDescriptor::RICIDescriptor* descriptors,
        ShapeDescriptor::RICIDescriptor* correspondingDescriptors,
        int* distances) {
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    int distanceScore;
    size_t needleImageIndex = blockIdx.x;
    int needleSquaredSum = computeImageSquaredSumGPU(descriptors[needleImageIndex]);
    bool needleImageIsConstant = needleSquaredSum == 0;

    if(!needleImageIsConstant) {
        distanceScore = compareRadialIntersectionCountImagePairGPU(
                descriptors, needleImageIndex,
                correspondingDescriptors, needleImageIndex);
    } else {
        distanceScore = compareConstantRadialIntersectionCountImagePairGPU(
                descriptors, needleImageIndex,
                correspondingDescriptors, needleImageIndex);
    }

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = distanceScore;
    }
}
#endif

ShapeDescriptor::cpu::array<int> ShapeDescriptor::computeRICIElementWiseModifiedSquareSumDistances(
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_descriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_correspondingDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<int> device_distances(device_descriptors.length);

    computeElementWiseRICIDistances<<<device_descriptors.length, 32>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            device_distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<int> resultDistances = device_distances.copyToCPU();
    ShapeDescriptor::free(device_distances);

    return resultDistances;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}