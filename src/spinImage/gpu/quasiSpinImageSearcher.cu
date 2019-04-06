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
    for(int row = laneIndex; row < spinImageWidthPixels; row++) {

        quasiSpinImagePixelType currentNeedlePixelValue =
                needleImages[needleImageIndex * spinImageElementCount + (row * spinImageWidthPixels + laneIndex)];
        quasiSpinImagePixelType currentHaystackPixelValue =
                haystackImages[haystackImageIndex * spinImageElementCount + (row * spinImageWidthPixels + laneIndex)];

        for(int col = laneIndex; col < spinImageWidthPixels - 32; col += 32) {
            quasiSpinImagePixelType nextNeedlePixelValue =
                    needleImages[needleImageIndex * spinImageElementCount + (row * spinImageWidthPixels + col)];
            quasiSpinImagePixelType nextHaystackPixelValue =
                    haystackImages[haystackImageIndex * spinImageElementCount + (row * spinImageWidthPixels + col)];

            quasiSpinImagePixelType nextRankNeedlePixelValue = __shfl_sync(0xFFFFFFFF,
                    // Input value
                                                                           (laneIndex == 0 ? nextNeedlePixelValue : currentNeedlePixelValue),
                    // Target thread
                                                                           (laneIndex == 31 ? 0 : threadIdx.x + 1));
            quasiSpinImagePixelType nextRankHaystackPixelValue = __shfl_sync(0xFFFFFFFF,
                    // Input value
                                                                             (laneIndex == 0 ? nextHaystackPixelValue : currentHaystackPixelValue),
                    // Target thread
                                                                             (laneIndex == 31 ? 0 : threadIdx.x + 1));

            quasiSpinImagePixelType needleDelta = nextRankNeedlePixelValue - currentNeedlePixelValue;
            quasiSpinImagePixelType haystackDelta = nextRankHaystackPixelValue - currentHaystackPixelValue;

            if(needleDelta != 0) {
                threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
            }

            currentNeedlePixelValue = nextNeedlePixelValue;
            currentHaystackPixelValue = nextHaystackPixelValue;
        }

        quasiSpinImagePixelType nextRankNeedlePixelValue = __shfl_sync(0xFFFFFFFF, currentNeedlePixelValue, laneIndex + 1);
        quasiSpinImagePixelType nextRankHaystackPixelValue = __shfl_sync(0xFFFFFFFF, currentHaystackPixelValue, laneIndex + 1);

        quasiSpinImagePixelType needleDelta = nextRankNeedlePixelValue - currentNeedlePixelValue;
        quasiSpinImagePixelType haystackDelta = nextRankHaystackPixelValue - currentHaystackPixelValue;

        threadScore += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
    }

    int imageScore = warpAllReduceSum(threadScore);

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


array<unsigned int> SpinImage::gpu::computeSearchResultRanks(
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