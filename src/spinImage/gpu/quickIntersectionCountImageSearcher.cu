#include "quickIntersectionCountImageSearcher.cuh"
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvidia/helper_cuda.h>

#ifndef warpSize
#define warpSize 32
#endif

const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;

__inline__ __device__ int warpAllReduceSum(int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ unsigned int getChunkAt(const unsigned int* imageArray, const size_t imageIndex, const int chunkIndex) {
    return imageArray[imageIndex * uintsPerQUICCImage + chunkIndex];
}

const int indexBasedWarpCount = 16;

__device__ int computeImageSumGPU(
        const unsigned int* needleImages,
        const size_t imageIndex) {

    const int laneIndex = threadIdx.x % 32;

    unsigned int threadSquaredSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0);

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, imageIndex, chunk);
        threadSquaredSum += __popc(needleChunk);
    }

    int squaredSum = warpAllReduceSum(threadSquaredSum);

    return squaredSum;
}

// TODO: early exit on this one too?
__device__ unsigned int compareConstantQUICCImagePairGPU(
        const unsigned int* haystackIncreasingImages,
        const unsigned int* haystackDecreasingImages,
        const size_t haystackImageIndex) {

    const int laneIndex = threadIdx.x % 32;

    unsigned int threadDeltaSquaredSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0);

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int haystackIncreasingChunk =
                getChunkAt(haystackIncreasingImages, haystackImageIndex, chunk);
        unsigned int haystackDecreasingChunk =
                getChunkAt(haystackDecreasingImages, haystackImageIndex, chunk);

        // Constant image is empty. Hence we only need to look at the haystack side of things.
        threadDeltaSquaredSum +=
                __popc(haystackIncreasingChunk) +
                __popc(haystackDecreasingChunk);
    }

    // image is constant.
    // In those situations, imageScore would always be 0
    // So we use an unfiltered squared sum instead
    unsigned int imageScore = warpAllReduceSum(threadDeltaSquaredSum);

    return imageScore;
}







__device__ int compareQUICCImagePairGPU(
        const unsigned int* needleIncreasingImages,
        const unsigned int* needleDecreasingImages,
        const size_t needleImageIndex,
        const unsigned int* haystackIncreasingImages,
        const unsigned int* haystackDecreasingImages,
        const size_t haystackImageIndex,
        const int distanceToBeat = INT_MAX) {

    int threadScore = 0;
    const int laneIndex = threadIdx.x % 32;

    static_assert(spinImageWidthPixels % 32 == 0);

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleIncreasingChunk = getChunkAt(needleIncreasingImages, needleImageIndex, chunk);
        unsigned int needleDecreasingChunk = getChunkAt(needleDecreasingImages, needleImageIndex, chunk);
        unsigned int haystackIncreasingChunk = getChunkAt(haystackIncreasingImages, haystackImageIndex, chunk);
        unsigned int haystackDecreasingChunk = getChunkAt(haystackDecreasingImages, haystackImageIndex, chunk);

        threadScore += __popc((needleIncreasingChunk ^ haystackIncreasingChunk) & needleIncreasingChunk);
        threadScore += __popc((needleDecreasingChunk ^ haystackDecreasingChunk) & needleDecreasingChunk);

#if ENABLE_RICI_COMPARISON_EARLY_EXIT
        int intermediateDistance = warpAllReduceSum(threadScore);
        if(intermediateDistance >= distanceToBeat) {
            return intermediateDistance;
        }
#endif
    }

    int imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}

__global__ void computeQUICCISearchResultIndices(
        const unsigned int* needleIncreasingDescriptors,
        const unsigned int* needleDecreasingDescriptors,
        const unsigned int* haystackIncreasingDescriptors,
        const unsigned int* haystackDecreasingDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ unsigned int referenceIncreasingImage[uintsPerQUICCImage];
    __shared__ unsigned int referenceDecreasingImage[uintsPerQUICCImage];

    for(unsigned int chunk = threadIdx.x; chunk < uintsPerQUICCImage; chunk += blockDim.x) {
        referenceIncreasingImage[chunk] = getChunkAt(needleIncreasingDescriptors, needleImageIndex, chunk);
        referenceDecreasingImage[chunk] = getChunkAt(needleDecreasingDescriptors, needleImageIndex, chunk);
    }

    __syncthreads();

    int referenceScore;

    int needleSquaredSum =
            computeImageSumGPU(referenceIncreasingImage, 0) +
            computeImageSumGPU(referenceDecreasingImage, 0);

    bool needleImageIsConstant = needleSquaredSum == 0;

    if(!needleImageIsConstant) {
        referenceScore = compareQUICCImagePairGPU(
                referenceIncreasingImage, referenceDecreasingImage, 0,
                haystackIncreasingDescriptors, haystackDecreasingDescriptors, needleImageIndex);
    } else {
        referenceScore = compareConstantQUICCImagePairGPU(
                haystackIncreasingDescriptors, haystackDecreasingDescriptors, needleImageIndex);
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
            pairScore = compareQUICCImagePairGPU(
                    referenceIncreasingImage, referenceDecreasingImage, 0,
                    haystackIncreasingDescriptors, haystackDecreasingDescriptors, haystackImageIndex,
                    referenceScore);
        } else {
            // If the image is constant, we use sum of squares as a fallback
            pairScore = compareConstantQUICCImagePairGPU(
                    haystackIncreasingDescriptors, haystackDecreasingDescriptors, haystackImageIndex);
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

SpinImage::array<unsigned int> SpinImage::gpu::computeQUICCImageSearchResultRanks(
        SpinImage::gpu::QUICCIImages device_needleDescriptors,
        size_t needleImageCount,
        SpinImage::gpu::QUICCIImages device_haystackDescriptors,
        size_t haystackImageCount,
        SpinImage::debug::QUICCISearchRunInfo* runInfo) {
    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = needleImageCount * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeQUICCISearchResultIndices << < needleImageCount, 32 * indexBasedWarpCount >> > (
            device_needleDescriptors.horizontallyIncreasingImages,
            device_needleDescriptors.horizontallyDecreasingImages,
            device_haystackDescriptors.horizontallyIncreasingImages,
            device_haystackDescriptors.horizontallyDecreasingImages,
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

__global__ void generateSearchResults(unsigned int* needleIncreasingDescriptors,
                                      unsigned int* needleDecreasingDescriptors,
                                      size_t needleImageCount,
                                      unsigned int* haystackIncreasingDescriptors,
                                      unsigned int* haystackDecreasingDescriptors,
                                      size_t haystackImageCount,
                                      SpinImage::gpu::QUICCISearchResults* searchResults) {

    size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

    if(needleImageIndex >= needleImageCount) {
        return;
    }

    static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
    size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
    int threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};

    const int blockCount = (SEARCH_RESULT_COUNT / 32);

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
        int score = compareQUICCImagePairGPU(
                needleIncreasingDescriptors,
                needleDecreasingDescriptors,
                needleImageIndex,
                haystackIncreasingDescriptors,
                haystackDecreasingDescriptors,
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

SpinImage::array<SpinImage::gpu::QUICCISearchResults> SpinImage::gpu::findQUICCImagesInHaystack(
        SpinImage::gpu::QUICCIImages device_needleDescriptors,
        size_t needleImageCount,
        SpinImage::gpu::QUICCIImages device_haystackDescriptors,
        size_t haystackImageCount) {
    size_t searchResultBufferSize = needleImageCount * sizeof(SpinImage::gpu::QUICCISearchResults);
    SpinImage::gpu::QUICCISearchResults* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

    std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(needleImageCount / warpCount) + 1, 32 * warpCount>>>(
            device_needleDescriptors.horizontallyIncreasingImages,
            device_needleDescriptors.horizontallyDecreasingImages,
            needleImageCount,
            device_haystackDescriptors.horizontallyIncreasingImages,
            device_haystackDescriptors.horizontallyDecreasingImages,
            haystackImageCount,
            device_searchResults);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

    array<SpinImage::gpu::QUICCISearchResults> searchResults;
    searchResults.content = new SpinImage::gpu::QUICCISearchResults[needleImageCount];
    searchResults.length = needleImageCount;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    return searchResults;
}
