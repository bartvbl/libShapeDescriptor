#include <algorithm>
#include "3dShapeContextSearcher.cuh"


const unsigned int warpCount = 16;


__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__device__ float compute3DSCPairCorrelationGPU(
        shapeContextBinType* descriptors,
        shapeContextBinType* otherDescriptors,
        size_t descriptorIndex,
        size_t otherDescriptorIndex) {

    for

    return correlation;
}

__global__ void calculateImageAverages(spinImagePixelType* images, float* averages) {
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
        float pixelValue = float(images[imageIndex * spinImageElementCount + x]);
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

__global__ void generateSearchResults(spinImagePixelType* needleDescriptors,
                                      size_t needleImageCount,
                                      spinImagePixelType* haystackDescriptors,
                                      size_t haystackImageCount,
                                      SpinImage::gpu::SpinImageSearchResults* searchResults,
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

                threadSearchResultScores[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultScores[startBlock], targetThread);
                threadSearchResultImageIndexes[startBlock] = __shfl_sync(0xFFFFFFFF, threadSearchResultImageIndexes[startBlock], targetThread);

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
        searchResults[needleImageIndex].resultIndices[block * 32 + laneID] = threadSearchResultImageIndexes[block];
        searchResults[needleImageIndex].resultScores[block * 32 + laneID] = threadSearchResultScores[block];
    }

}

SpinImage::array<unsigned int> SpinImage::gpu::compute3DSCSearchResultRanks(
        array<shapeContextBinType> device_needleDescriptors,
        size_t needleDescriptorCount,
        array<shapeContextBinType> device_haystackDescriptors,
        size_t haystackDescriptorCount,
        SpinImage::debug::SCSearchRunInfo* runInfo) {

    // Step 1: Compute image averages, since they're constant and are needed for each comparison

    float* device_needleImageAverages;
    float* device_haystackImageAverages;
    checkCudaErrors(cudaMalloc(&device_needleImageAverages, needleImageCount * sizeof(float)));
    checkCudaErrors(cudaMalloc(&device_haystackImageAverages, haystackImageCount * sizeof(float)));

    std::cout << "\t\tComputing image averages.." << std::endl;
    calculateImageAverages<<<needleImageCount, 32>>>(device_needleDescriptors.content, device_needleImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());
    calculateImageAverages<<<haystackImageCount, 32>>>(device_haystackDescriptors.content, device_haystackImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());

    // Step 2: Perform search

    size_t searchResultBufferSize = needleImageCount * sizeof(SpinImageSearchResults);
    SpinImageSearchResults* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));

    std::cout << "\t\tPerforming search.." << std::endl;
    auto start = std::chrono::steady_clock::now();

    generateSearchResults<<<(needleImageCount / warpCount) + 1, 32 * warpCount>>>(
            device_needleDescriptors.content,
                    needleImageCount,
                    device_haystackDescriptors.content,
                    haystackImageCount,
                    device_searchResults,
                    device_needleImageAverages,
                    device_haystackImageAverages);
    checkCudaErrors(cudaDeviceSynchronize());

    std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
    std::cout << "\t\t\tExecution time: " << duration.count() << std::endl;

    // Step 3: Copying results to CPU

    array<SpinImageSearchResults> searchResults;
    searchResults.content = new SpinImageSearchResults[needleImageCount];
    searchResults.length = needleImageCount;

    checkCudaErrors(cudaMemcpy(searchResults.content, device_searchResults, searchResultBufferSize, cudaMemcpyDeviceToHost));

    // Cleanup

    cudaFree(device_needleImageAverages);
    cudaFree(device_haystackImageAverages);
    cudaFree(device_searchResults);

    return searchResults;



    SpinImage::array<unsigned int> searchResultIndices = {
            needleDescriptorCount,
            new unsigned int[needleDescriptorCount]
    };
    std::fill(searchResultIndices.content, searchResultIndices.content + needleDescriptorCount, 0);
    return searchResultIndices;
}