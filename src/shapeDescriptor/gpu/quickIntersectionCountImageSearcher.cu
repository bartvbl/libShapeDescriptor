#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#endif

#include <shapeDescriptor/shapeDescriptor.h>
#include <chrono>
#include <iostream>
#include <shapeDescriptor/utilities/weightedHamming.cuh>
#include <cfloat>

#ifndef warpSize
#define warpSize 32
#endif


#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
const unsigned int uintsPerQUICCImage = (spinImageWidthPixels * spinImageWidthPixels) / 32;



__inline__ __device__ unsigned int getChunkAt(const ShapeDescriptor::QUICCIDescriptor* image, const size_t imageIndex, const int chunkIndex) {
    return image[imageIndex].contents[chunkIndex];
}

const int indexBasedWarpCount = 16;

__device__ int computeImageSumGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t imageIndex) {

    const int laneIndex = threadIdx.x % 32;

    unsigned int threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes images are multiples of warp size wide");

    for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
        unsigned int needleChunk = getChunkAt(needleImages, imageIndex, chunk);
        threadSum += __popc(needleChunk);
    }

    int sum = ShapeDescriptor::warpAllReduceSum(threadSum);

    return sum;
}

__device__ ShapeDescriptor::quicciDistanceType compareConstantQUICCImagePairGPU(
        const ShapeDescriptor::QUICCIDescriptor* haystackImages,
        const size_t haystackImageIndex) {

    const int laneIndex = threadIdx.x % 32;

    ShapeDescriptor::quicciDistanceType threadSum = 0;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

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

    return ShapeDescriptor::warpAllReduceSum(threadSum);
}

__device__ ShapeDescriptor::quicciDistanceType compareQUICCImagePairGPU(
        const ShapeDescriptor::QUICCIDescriptor* needleImages,
        const size_t needleImageIndex,
        const ShapeDescriptor::QUICCIDescriptor* haystackImages,
        const size_t haystackImageIndex
#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
        , ShapeDescriptor::utilities::HammingWeights hammingWeights
#endif
        ) {


    const int laneIndex = threadIdx.x % 32;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    ShapeDescriptor::quicciDistanceType threadScore = 0;

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

    ShapeDescriptor::quicciDistanceType imageScore = ShapeDescriptor::warpAllReduceSum(threadScore);

    return imageScore;
}

__global__ void computeQUICCISearchResultIndices(
        const ShapeDescriptor::QUICCIDescriptor* needleDescriptors,
        const ShapeDescriptor::QUICCIDescriptor* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ ShapeDescriptor::QUICCIDescriptor referenceImage;

    for(unsigned int chunk = threadIdx.x; chunk < uintsPerQUICCImage; chunk += blockDim.x) {
        referenceImage.contents[chunk] = getChunkAt(needleDescriptors, needleImageIndex, chunk);
    }

    __syncthreads();

    ShapeDescriptor::quicciDistanceType referenceScore;

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
        ShapeDescriptor::quicciDistanceType pairScore;
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
#endif

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::computeQUICCImageSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors,
        ShapeDescriptor::QUICCISearchExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeQUICCISearchResultIndices <<< device_needleDescriptors.length, 32 * indexBasedWarpCount >>> (
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
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}








#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeElementWiseQUICCIDistances(
        ShapeDescriptor::QUICCIDescriptor* descriptors,
        ShapeDescriptor::QUICCIDescriptor* correspondingDescriptors,
        ShapeDescriptor::QUICCIDistances* distances) {
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;
    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    ShapeDescriptor::QUICCIDistances imageDistances;

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

    imageDistances.clutterResistantDistance = ShapeDescriptor::warpAllReduceSum(threadClutterResistantDistance);
    imageDistances.hammingDistance = ShapeDescriptor::warpAllReduceSum(threadHammingDistance);
    imageDistances.weightedHammingDistance = ShapeDescriptor::warpAllReduceSum(threadWeightedHammingDistance);
    imageDistances.needleImageBitCount = referenceImageBitCount;

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = imageDistances;
    }
}
#endif

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void computeElementWiseQUICCIWeightedHammingDistances(
        ShapeDescriptor::QUICCIDescriptor* descriptors,
        ShapeDescriptor::QUICCIDescriptor* correspondingDescriptors,
        float* distances) {
    const size_t descriptorIndex = blockIdx.x;
    const int laneIndex = threadIdx.x;
    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    int referenceImageBitCount = computeImageSumGPU(descriptors, descriptorIndex);

    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeights(referenceImageBitCount, spinImageWidthPixels * spinImageWidthPixels);

    bool needleImageIsConstant = referenceImageBitCount == 0;

    float threadWeightedHammingDistance = 0;

    if(!needleImageIsConstant) {
        for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
            unsigned int needleChunk = getChunkAt(descriptors, descriptorIndex, chunk);
            unsigned int haystackChunk = getChunkAt(correspondingDescriptors, descriptorIndex, chunk);
            threadWeightedHammingDistance += ShapeDescriptor::utilities::computeChunkWeightedHammingDistance(hammingWeights, needleChunk, haystackChunk);
        }
    } else {
        for (int chunk = laneIndex; chunk < uintsPerQUICCImage; chunk += warpSize) {
            unsigned int haystackChunk =
                    getChunkAt(correspondingDescriptors, descriptorIndex, chunk);

            // Constant image is empty. Hence we only need to look at the haystack side of things.
            // Since a constant needle image will always use this function for ranking, and due to avoiding zero
            // division errors the weight of a missed unset bit is always 1, we can use the same ranking function
            // for weighted hamming as the other ranking functions.
            threadWeightedHammingDistance += float(__popc(haystackChunk));
        }
    }

    float weightedHammingDistance = ShapeDescriptor::warpAllReduceSum(threadWeightedHammingDistance);

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = weightedHammingDistance;
    }
}
#endif

ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDistances>
ShapeDescriptor::computeQUICCIElementWiseDistances(ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_descriptors,
                                                  ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_correspondingDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDistances> distances(device_descriptors.length);

    computeElementWiseQUICCIDistances<<<device_descriptors.length, 32>>>(
        device_descriptors.content,
        device_correspondingDescriptors.content,
        distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDistances> resultDistances = distances.copyToCPU();
    ShapeDescriptor::free(distances);

    return resultDistances;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::cpu::array<float> ShapeDescriptor::computeQUICCIElementWiseWeightedHammingDistances(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_descriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_correspondingDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<float> distances(device_descriptors.length);

    computeElementWiseQUICCIWeightedHammingDistances<<<device_descriptors.length, 32>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<float> resultDistances = distances.copyToCPU();
    ShapeDescriptor::free(distances);

    return resultDistances;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}



















const unsigned int warpCount = 16;
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void generateSearchResults(ShapeDescriptor::QUICCIDescriptor* needleDescriptors,
                                      size_t needleImageCount,
                                      ShapeDescriptor::QUICCIDescriptor* haystackDescriptors,
                                      size_t haystackImageCount,
                                      ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>* searchResults) {

    size_t needleImageIndex = warpCount * blockIdx.x + (threadIdx.x / 32);

    if(needleImageIndex >= needleImageCount) {
        return;
    }

    static_assert(SEARCH_RESULT_COUNT == 128, "Array initialisation needs to change if search result count is changed");
    size_t threadSearchResultImageIndexes[SEARCH_RESULT_COUNT / 32] = {UINT_MAX, UINT_MAX, UINT_MAX, UINT_MAX};
#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
    ShapeDescriptor::gpu::quicciDistanceType threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    ShapeDescriptor::utilities::HammingWeights hammingWeights = ShapeDescriptor::utilities::computeWeightedHammingWeightsGPU(needleDescriptors[needleImageIndex]);
#else
    ShapeDescriptor::quicciDistanceType threadSearchResultScores[SEARCH_RESULT_COUNT / 32] = {INT_MAX, INT_MAX, INT_MAX, INT_MAX};
#endif
    const int blockCount = (SEARCH_RESULT_COUNT / 32);

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackImageCount; haystackImageIndex++) {
        ShapeDescriptor::quicciDistanceType score = compareQUICCImagePairGPU(
                needleDescriptors,
                needleImageIndex,
                haystackDescriptors,
                haystackImageIndex
#if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
                , hammingWeights
#endif
                );

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

ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>> ShapeDescriptor::findQUICCImagesInHaystack(
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>);
    ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>* device_searchResults;
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

    ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>> searchResults;
    searchResults.content = new ShapeDescriptor::SearchResults<ShapeDescriptor::quicciDistanceType>[device_needleDescriptors.length];
    searchResults.length = device_needleDescriptors.length;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_searchResults);

    return searchResults;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}