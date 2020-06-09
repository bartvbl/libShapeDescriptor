#include "quickIntersectionCountImageSearcher.cuh"
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvidia/helper_cuda.h>
#include <spinImage/utilities/weightedHamming.cuh>

#ifndef warpSize
#define warpSize 32
#endif

#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    typedef float distanceType;
#else
    typedef unsigned int distanceType;
#endif

const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;

__inline__ __device__ distanceType warpAllReduceSum(distanceType val) {
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

    unsigned int threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0);

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, imageIndex, chunk);
        threadSum += __popc(needleChunk);
    }

    int sum = warpAllReduceSum(threadSum);

    return sum;
}

__device__ distanceType compareConstantQUICCImagePairGPU(
        const unsigned int* haystackImages,
        const size_t haystackImageIndex) {

    const int laneIndex = threadIdx.x % 32;

    distanceType threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0);

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int haystackChunk =
                getChunkAt(haystackImages, haystackImageIndex, chunk);

        // Constant image is empty. Hence we only need to look at the haystack side of things.
#if QUICCI_DISTANCE_FUNCTION == CLUTTER_RESISTANT_DISTANCE
        threadSum += __popc(haystackChunk);
#elif QUICCI_DISTANCE_FUNCTION == HAMMING_DISTANCE
        threadSum += __popc(haystackChunk);
#elif QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
        // Since a constant needle image will always use this function for ranking, and due to avoiding zero
        // division errors the weight of a missed unset bit is always 1, we can use the same ranking function
        // for weighted hamming as the other ranking functions.
        threadSum += float(__popc(haystackChunk));
#endif
    }

    return warpAllReduceSum(threadSum);
}

__device__ distanceType compareQUICCImagePairGPU(
        const unsigned int* needleImages,
        const size_t needleImageIndex,
        const unsigned int* haystackImages,
        const size_t haystackImageIndex
#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
        , SpinImage::utilities::HammingWeights hammingWeights
#endif
        ) {


    const int laneIndex = threadIdx.x % 32;

    static_assert(spinImageWidthPixels % 32 == 0);

    distanceType threadScore = 0;

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, needleImageIndex, chunk);
        unsigned int haystackChunk = getChunkAt(haystackImages, haystackImageIndex, chunk);

#if QUICCI_DISTANCE_FUNCTION == CLUTTER_RESISTANT_DISTANCE
        threadScore += __popc((needleChunk ^ haystackChunk) & needleChunk);
#elif QUICCI_DISTANCE_FUNCTION == HAMMING_DISTANCE
        threadScore += __popc(needleChunk ^ haystackChunk);
#elif QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
        threadScore += SpinImage::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
#endif
    }

    distanceType imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}

__global__ void computeQUICCISearchResultIndices(
        const unsigned int* needleDescriptors,
        const unsigned int* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ unsigned int referenceImage[uintsPerQUICCImage];

    for(unsigned int chunk = threadIdx.x; chunk < uintsPerQUICCImage; chunk += blockDim.x) {
        referenceImage[chunk] = getChunkAt(needleDescriptors, needleImageIndex, chunk);
    }

    __syncthreads();

    distanceType referenceScore;

    int referenceImageBitCount = computeImageSumGPU(referenceImage, 0);

#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    SpinImage::utilities::HammingWeights hammingWeights = SpinImage::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);
#endif

    bool needleImageIsConstant = referenceImageBitCount == 0;

    if(!needleImageIsConstant) {
        referenceScore = compareQUICCImagePairGPU(
               referenceImage, 0,
               haystackDescriptors, needleImageIndex
               #if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
               , hammingWeights
               #endif
               );
    } else {
        referenceScore = compareConstantQUICCImagePairGPU(haystackDescriptors, needleImageIndex);
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
        distanceType pairScore;
        if(!needleImageIsConstant) {
            // If there's variation in the image, we'll use the regular distance function
            pairScore = compareQUICCImagePairGPU(
                    referenceImage, 0,
                    haystackDescriptors, haystackImageIndex
                    #if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
                    , hammingWeights
                    #endif
                    );
        } else {
            // If the image is constant, we use sum of squares as a fallback
            pairScore = compareConstantQUICCImagePairGPU(haystackDescriptors, haystackImageIndex);
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
            device_needleDescriptors.images,
            device_haystackDescriptors.images,
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