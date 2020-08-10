#include "quickIntersectionCountImageSearcher.cuh"
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <chrono>
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvidia/helper_cuda.h>
#include <shapeDescriptor/utilities/weightedHamming.cuh>

#ifndef warpSize
#define warpSize 32
#endif

#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    typedef float distanceType;
#else
    typedef unsigned int distanceType;
#endif

const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;

__inline__ __device__ unsigned int warpAllReduceSum(unsigned int val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ unsigned int getChunkAt(const ShapeDescriptor::gpu::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex) {
    return image[imageIndex].contents[chunkIndex];
}

const int indexBasedWarpCount = 16;

__device__ int computeImageSumGPU(
        const ShapeDescriptor::gpu::QUICCIDescriptor* needleImages,
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
        const ShapeDescriptor::gpu::QUICCIDescriptor* haystackImages,
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
        const ShapeDescriptor::gpu::QUICCIDescriptor* needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::gpu::QUICCIDescriptor* haystackImages,
        const size_t haystackImageIndex
#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
        , ShapeDescriptor::utilities::HammingWeights hammingWeights
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
        threadScore += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
#endif
    }

    distanceType imageScore = warpAllReduceSum(threadScore);

    return imageScore;
}

__global__ void computeQUICCISearchResultIndices(
        const ShapeDescriptor::gpu::QUICCIDescriptor* needleDescriptors,
        const ShapeDescriptor::gpu::QUICCIDescriptor* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ ShapeDescriptor::gpu::QUICCIDescriptor referenceImage;

    for(unsigned int chunk = threadIdx.x; chunk < uintsPerQUICCImage; chunk += blockDim.x) {
        referenceImage.contents[chunk] = getChunkAt(needleDescriptors, needleImageIndex, chunk);
    }

    __syncthreads();

    distanceType referenceScore;

    int referenceImageBitCount = computeImageSumGPU(&referenceImage, 0);

#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);
#endif

    bool needleImageIsConstant = referenceImageBitCount == 0;

    if(!needleImageIsConstant) {
        referenceScore = compareQUICCImagePairGPU(
               &referenceImage, 0,
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
                    &referenceImage, 0,
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

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::gpu::computeQUICCImageSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::QUICCIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::QUICCIDescriptor> device_haystackDescriptors,
        ShapeDescriptor::debug::QUICCISearchExecutionTimes* executionTimes) {
    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeQUICCISearchResultIndices << < device_needleDescriptors.length, 32 * indexBasedWarpCount >> > (
            device_needleDescriptors.content,
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
}









__global__ void computeElementWiseQUICCIDistances(
        ShapeDescriptor::gpu::QUICCIDescriptor* descriptors,
        ShapeDescriptor::gpu::QUICCIDescriptor* correspondingDescriptors,
        ShapeDescriptor::gpu::QUICCIDistances* distances) {
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;
    static_assert(spinImageWidthPixels % 32 == 0);

    ShapeDescriptor::gpu::QUICCIDistances imageDistances;

    int referenceImageBitCount = computeImageSumGPU(descriptors, descriptorIndex);

    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

    bool needleImageIsConstant = referenceImageBitCount == 0;

    unsigned int threadClutterResistantDistance = 0;
    unsigned int threadHammingDistance = 0;
    float threadWeightedHammingDistance = 0;

    if(!needleImageIsConstant) {
        for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
            unsigned int needleChunk = getChunkAt(descriptors, descriptorIndex, chunk);
            unsigned int haystackChunk = getChunkAt(correspondingDescriptors, descriptorIndex, chunk);

            threadClutterResistantDistance += __popc((needleChunk ^ haystackChunk) & needleChunk);
            threadHammingDistance += __popc(needleChunk ^ haystackChunk);
            threadWeightedHammingDistance += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
        }
    } else {
        for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
            unsigned int haystackChunk =
                    getChunkAt(correspondingDescriptors, descriptorIndex, chunk);

            // Constant image is empty. Hence we only need to look at the haystack side of things.
            threadClutterResistantDistance += __popc(haystackChunk);
            threadHammingDistance += __popc(haystackChunk);

            // Since a constant needle image will always use this function for ranking, and due to avoiding zero
            // division errors the weight of a missed unset bit is always 1, we can use the same ranking function
            // for weighted hamming as the other ranking functions.
            threadWeightedHammingDistance += float(__popc(haystackChunk));
        }
    }

    imageDistances.clutterResistantDistance = warpAllReduceSum(threadClutterResistantDistance);
    imageDistances.hammingDistance = warpAllReduceSum(threadHammingDistance);
    imageDistances.weightedHammingDistance = warpAllReduceSum(threadWeightedHammingDistance);
    imageDistances.needleImageBitCount = referenceImageBitCount;

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = imageDistances;
    }
}


ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances>
ShapeDescriptor::gpu::computeQUICCIElementWiseDistances(ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::QUICCIDescriptor> device_descriptors,
                                                  ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::QUICCIDescriptor> device_correspondingDescriptors) {
    size_t searchResultBufferSize = device_descriptors.length * sizeof(ShapeDescriptor::gpu::QUICCIDistances);
    ShapeDescriptor::gpu::QUICCIDistances* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    computeElementWiseQUICCIDistances<<<device_descriptors.length, 32>>>(
        device_descriptors.content,
        device_correspondingDescriptors.content,
        device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> resultDistances;
    resultDistances.content = new ShapeDescriptor::gpu::QUICCIDistances[device_descriptors.length];
    resultDistances.length = device_descriptors.length;

    checkCudaErrors(cudaMemcpy(resultDistances.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(device_searchResults);

    return resultDistances;
}
