#include <algorithm>
#include <chrono>
#include "3dShapeContextSearcher.cuh"
#include <nvidia/helper_cuda.h>


const int indexBasedWarpCount = 16;

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;


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



    return 0;
}

__global__ void computeShapeContextSearchResultIndices(
        shapeContextBinType* needleDescriptors,
        shapeContextBinType* haystackDescriptors,
        size_t haystackImageCount,
        unsigned int* searchResults) {
    size_t needleImageIndex = blockIdx.x;

    __shared__ shapeContextBinType referenceDescriptor[elementsPerShapeContextDescriptor];
    for(unsigned int index = threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x) {
        referenceDescriptor[index] = needleDescriptors[elementsPerShapeContextDescriptor * needleImageIndex + index];
    }

    __syncthreads();

    float referenceDistance = compute3DSCPairCorrelationGPU(
            referenceDescriptor,
            haystackDescriptors,
            0,
            needleImageIndex);

    // No image pair can have a better distance than 0, so we can just stop the search right here
    if(referenceDistance == 0) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackDescriptorIndex = threadIdx.x / 32; haystackDescriptorIndex < haystackImageCount; haystackDescriptorIndex += indexBasedWarpCount) {
        if(needleImageIndex == haystackDescriptorIndex) {
            continue;
        }

        float distance = compute3DSCPairCorrelationGPU(
                referenceDescriptor,
                haystackDescriptors,
                0,
                haystackDescriptorIndex);

        // We've found a result that's better than the reference one. That means this search result would end up
        // above ours in the search result list. We therefore move our search result down by 1.
        if(distance < referenceDistance) {
            searchResultRank++;
        }
    }

    // Since we're running multiple warps, we need to add all indices together to get the correct ranks
    if(threadIdx.x % 32 == 0) {
        atomicAdd(&searchResults[needleImageIndex], searchResultRank);
    }
}



SpinImage::array<unsigned int> SpinImage::gpu::compute3DSCSearchResultRanks(
        array<shapeContextBinType> device_needleDescriptors,
        size_t needleDescriptorCount,
        array<shapeContextBinType> device_haystackDescriptors,
        size_t haystackDescriptorCount,
        SpinImage::debug::SCSearchRunInfo* runInfo) {

    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = needleDescriptorCount * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeShapeContextSearchResultIndices<<<needleDescriptorCount, 32 * indexBasedWarpCount>>>(
        device_needleDescriptors.content,
        device_haystackDescriptors.content,
        haystackDescriptorCount,
        device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

    array<unsigned int> resultIndices;
    resultIndices.content = new unsigned int[needleDescriptorCount];
    resultIndices.length = needleDescriptorCount;

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