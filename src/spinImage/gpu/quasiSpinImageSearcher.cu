#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/libraryBuildSettings.h>
#include <cuda_runtime.h>
#include <curand_mtgp32_kernel.h>
#include <tgmath.h>
#include <assert.h>
#include <iostream>
#include <climits>
#include <cfloat>
#include <chrono>
#include <typeinfo>
#include "nvidia/helper_cuda.h"
#include "quasiSpinImageSearcher.cuh"

__inline__ __device__ int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

const int indexBasedWarpCount = 16;

__device__ int compareQuasiSpinImagePairGPU(
        const quasiSpinImagePixelType* needleImages,
        const size_t needleImageIndex,
        const quasiSpinImagePixelType* haystackImages,
        const size_t haystackImageIndex) {
    int threadScore = 0;
    const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
    const int laneIndex = threadIdx.x % 32;
    unsigned int threadSquaredSum = 0;
    unsigned int threadDeltaSquaredSum = 0;

    quasiSpinImagePixelType previousWarpLastNeedlePixelValue = 0;
    quasiSpinImagePixelType previousWarpLastHaystackPixelValue = 0;

    for(int pixel = laneIndex; pixel < spinImageWidthPixels * spinImageWidthPixels; pixel += warpSize) {
        quasiSpinImagePixelType currentNeedlePixelValue =
            needleImages[needleImageIndex * spinImageElementCount + pixel];
        quasiSpinImagePixelType currentHaystackPixelValue =
            haystackImages[haystackImageIndex * spinImageElementCount + pixel];

        int targetThread;
        if(laneIndex > 0) {
            targetThread = laneIndex - 1;
        } else if(pixel % spinImageWidthPixels != 0) {
            targetThread = 31;
        } else {
            targetThread = 0;
        }

        quasiSpinImagePixelType threadNeedleValue;
        quasiSpinImagePixelType threadHaystackValue;

        if(laneIndex == 31) {
            threadNeedleValue = previousWarpLastNeedlePixelValue;
            threadHaystackValue = previousWarpLastHaystackPixelValue;
        } else {
            threadNeedleValue = currentNeedlePixelValue;
            threadHaystackValue = currentHaystackPixelValue;
        }

        quasiSpinImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);
        quasiSpinImagePixelType previousHaystackPixelValue = __shfl_sync(0xFFFFFFFF, threadHaystackValue, targetThread);

        int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
        int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

        if(needleDelta != 0) {
            threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
        }

        if(laneIndex != 31) {
            int imageDelta = currentNeedlePixelValue - currentHaystackPixelValue;
            threadSquaredSum += unsigned(needleDelta * needleDelta);
            threadDeltaSquaredSum += unsigned(imageDelta * imageDelta);
        }

        // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
        previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
        previousWarpLastHaystackPixelValue = currentHaystackPixelValue;
    }

    int imageScore = warpAllReduceSum(threadScore);

    int squaredSum = warpAllReduceSum(threadSquaredSum);
    int deltaSquaredSum = warpAllReduceSum(threadDeltaSquaredSum);

    // image is constant
    if(squaredSum == 0) {
        imageScore = deltaSquaredSum;
    }

    return imageScore;
}

__global__ void computeQuasiSpinImageSearchResultIndices(
        quasiSpinImagePixelType* needleDescriptors,
        quasiSpinImagePixelType* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ quasiSpinImagePixelType referenceImage[spinImageWidthPixels * spinImageWidthPixels];
    for(unsigned int index = threadIdx.x; index < spinImageWidthPixels * spinImageWidthPixels; index += blockDim.x) {
        referenceImage[index] = needleDescriptors[spinImageWidthPixels * spinImageWidthPixels * needleImageIndex + index];
    }

    __syncthreads();

    int referenceScore = compareQuasiSpinImagePairGPU(referenceImage, 0, haystackDescriptors, needleImageIndex);

    if(referenceScore == 0) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackImageIndex = threadIdx.x / 32; haystackImageIndex < haystackImageCount; haystackImageIndex += indexBasedWarpCount) {
        if (needleImageIndex == haystackImageIndex) {
            continue;
        }

        int pairScore = compareQuasiSpinImagePairGPU(referenceImage, 0, haystackDescriptors, haystackImageIndex);

        if(pairScore < referenceScore) {
            searchResultRank++;
        }
    }

    // Since we're running multiple warps, we need to add all indices together to get the correct ranks
    if(threadIdx.x % 32 == 0) {
        atomicAdd(&searchResults[needleImageIndex], searchResultRank);
    }
}


array<unsigned int> SpinImage::gpu::computeQuasiSpinImageSearchResultRanks(
        array<quasiSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<quasiSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount,
        SpinImage::debug::QSISearchRunInfo* runInfo) {

    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = needleImageCount * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeQuasiSpinImageSearchResultIndices<<<needleImageCount, 32 * indexBasedWarpCount>>>(
            device_needleDescriptors.content,
                    device_haystackDescriptors.content,
                    haystackImageCount,
                    device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

    array<unsigned int> resultIndices;
    resultIndices.content = new unsigned int[needleImageCount];
    resultIndices.length = needleImageCount;

    checkCudaErrors(cudaMemcpy(resultIndices.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    std::chrono::milliseconds executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - executionStart);

    if(runInfo != nullptr) {
        runInfo->searchExecutionTimeSeconds = double(searchDuration.count()) / 1000.0;
        runInfo->totalExecutionTimeSeconds = double(executionDuration.count()) / 1000.0;
    }

    return resultIndices;
}































const unsigned int warpCount = 16;

__global__ void generateSearchResults(quasiSpinImagePixelType* needleDescriptors,
                                      size_t needleImageCount,
                                      quasiSpinImagePixelType* haystackDescriptors,
                                      size_t haystackImageCount,
                                      QuasiSpinImageSearchResults* searchResults) {

    size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

    if(needleImageIndex >= needleImageCount) {
        return;
    }

    static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
    size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    int threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

    const int blockCount = (SEARCH_RESULT_COUNT / 32);

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
        int score = compareQuasiSpinImagePairGPU(
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

                threadSearchResultScores[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[startBlock], targetThread);
                threadSearchResultImageIndexes[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[startBlock], targetThread);

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
        searchResults[needleImageIndex].resultIndices[block * 32 + laneID] = threadSearchResultImageIndexes[block];
        searchResults[needleImageIndex].resultScores[block * 32 + laneID] = threadSearchResultScores[block];
    }

}

array<QuasiSpinImageSearchResults> SpinImage::gpu::findQuasiSpinImagesInHaystack(
        array<quasiSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<quasiSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount) {

    size_t searchResultBufferSize = needleImageCount * sizeof(QuasiSpinImageSearchResults);
    QuasiSpinImageSearchResults* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

    std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(needleImageCount / warpCount) + 1, 32 * warpCount>>>(
            device_needleDescriptors.content,
            needleImageCount,
            device_haystackDescriptors.content,
            haystackImageCount,
            device_searchResults);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

    array<QuasiSpinImageSearchResults> searchResults;
    searchResults.content = new QuasiSpinImageSearchResults[needleImageCount];
    searchResults.length = needleImageCount;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    return searchResults;
}