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
        const size_t haystackImageIndex,
        const int distanceToBeat = INT_MAX) {
    int threadScore = 0;
    const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;
    const int laneIndex = threadIdx.x % 32;
    unsigned int threadSquaredSum = 0;
    unsigned int threadDeltaSquaredSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0);


    // Scores are computed one row at a time.
    // We differentiate between rows to ensure the final pixel of the previous row does not
    // affect the first pixel of the next one.
    for(int row = 0; row < spinImageWidthPixels; row++) {
        quasiSpinImagePixelType previousWarpLastNeedlePixelValue = 0;
        quasiSpinImagePixelType previousWarpLastHaystackPixelValue = 0;
        // Each thread processes one pixel, a warp processes therefore 32 pixels per iteration
        for (int pixel = laneIndex; pixel < spinImageWidthPixels; pixel += warpSize) {
            quasiSpinImagePixelType currentNeedlePixelValue =
                    needleImages[needleImageIndex * spinImageElementCount + row * spinImageWidthPixels + pixel];
            quasiSpinImagePixelType currentHaystackPixelValue =
                    haystackImages[haystackImageIndex * spinImageElementCount + row * spinImageWidthPixels + pixel];

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
            else if (pixel != 0) {
                // If pixel is not the first pixel in the row, we read the difference value from the previous iteration
                targetThread = 31;
            } else {
                // If the pixel is the leftmost pixel in the row, we give targetThread a dummy value that will always
                // result in a pixel delta of zero: our own thread with ID 0.
                targetThread = 0;
            }

            quasiSpinImagePixelType threadNeedleValue;
            quasiSpinImagePixelType threadHaystackValue;

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
            quasiSpinImagePixelType previousNeedlePixelValue = __shfl_sync(0xFFFFFFFF, threadNeedleValue, targetThread);
            quasiSpinImagePixelType previousHaystackPixelValue = __shfl_sync(0xFFFFFFFF, threadHaystackValue,
                                                                             targetThread);

            // The distance measure this function computes is based on deltas between pairs of pixels
            int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
            int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

            // This if statement makes a massive difference in the clutter resitant performance of this method
            // It only counts least squares differences if the needle image has a change in intersection count
            // Which is usually something very specific to that object.
            if (needleDelta != 0) {
                threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
            }

            // This bit handles the case where an image is completely constant.
            // In that case, we use the absolute sum of squares as a distance function instead
            if (laneIndex != 31) {
                int imageDelta = int(currentNeedlePixelValue) - int(currentHaystackPixelValue);
                threadSquaredSum += unsigned(needleDelta * needleDelta);
                threadDeltaSquaredSum += unsigned(imageDelta * imageDelta);
            }

            // This only matters for thread 31, so no need to broadcast it using a shuffle instruction
            previousWarpLastNeedlePixelValue = currentNeedlePixelValue;
            previousWarpLastHaystackPixelValue = currentHaystackPixelValue;
        }

        // At the end of each block of 8 rows, check whether we can do an early exit
        if(row % 2 == 1 && row != (spinImageWidthPixels - 1)) {
            int intermediateDistance = warpAllReduceSum(threadScore);
            if(intermediateDistance > distanceToBeat) {
                return intermediateDistance;
            }
        }
    }

    int imageScore = warpAllReduceSum(threadScore);

    int squaredSum = warpAllReduceSum(threadSquaredSum);

    // image is constant. 
    // In those situations, imageScore would always be 0
    // So we use an unfiltered squared sum instead
    if(squaredSum == 0) {
        int deltaSquaredSum = warpAllReduceSum(threadDeltaSquaredSum);  
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
        int pairScore = compareQuasiSpinImagePairGPU(referenceImage, 0, haystackDescriptors, haystackImageIndex, referenceScore);

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