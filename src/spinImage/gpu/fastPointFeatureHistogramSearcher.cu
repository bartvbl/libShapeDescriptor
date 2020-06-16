#include <iostream>
#include <chrono>
#include <cassert>
#include <nvidia/helper_cuda.h>
#include "fastPointFeatureHistogramSearcher.cuh"

/*__global__ void calculateSceneAverage(SpinImage::gpu::FPFHHistogram33* histograms, SpinImage::gpu::FPFHHistogram33* averageHistogram, unsigned int count) {
    // Using a running average for better numerical accuracy
    float average = 0;

    for(int i = 0; i < count; i++) {
        average = average + (histograms[i].contents[threadIdx.x] - average) / float(i + 1);
    }

    averageHistogram[threadIdx.x] = average;
}*/

__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float computeDescriptorAverage(SpinImage::gpu::FPFHHistogram33* descriptor) {
    float threadSum = 0;
    threadSum += descriptor->contents[threadIdx.x];
    if(threadIdx.x == 0) {
        threadSum += descriptor->contents[32];
    }
    float totalSum = warpAllReduceSum(threadSum);
    return totalSum / 33.0f;
}

__device__ float computeFPFHDescriptorSimilarity(
        SpinImage::gpu::FPFHHistogram33* needleDescriptor,
        float needleDescriptorAverage,
        SpinImage::gpu::FPFHHistogram33* haystackDescriptor) {

    float haystackDescriptorAverage = computeDescriptorAverage(haystackDescriptor);

    float threadSquaredSumX = 0;
    float threadSquaredSumY = 0;
    float threadMultiplicativeSum = 0;

    for(int i = threadIdx.x; i < 33; i += warpSize) {
        float needleDescriptorValue = needleDescriptor->contents[i];
        float haystackDescriptorValue = haystackDescriptor->contents[i];

        float deltaX = float(needleDescriptorValue) - needleDescriptorAverage;
        float deltaY = float(haystackDescriptorValue) - haystackDescriptorAverage;

        threadSquaredSumX += deltaX * deltaX;
        threadSquaredSumY += deltaY * deltaY;
        threadMultiplicativeSum += deltaX * deltaY;
    }

    float squaredSumX = float(sqrt(warpAllReduceSum(threadSquaredSumX)));
    float squaredSumY = float(sqrt(warpAllReduceSum(threadSquaredSumY)));
    float multiplicativeSum = warpAllReduceSum(threadMultiplicativeSum);

    float correlation = multiplicativeSum / (squaredSumX * squaredSumY);

    return correlation;
}


__global__ void computeFPFHSearchResultIndices(
        SpinImage::gpu::FPFHHistogram33* needleDescriptors,
        SpinImage::gpu::FPFHHistogram33* haystackDescriptors,
        size_t haystackDescriptorCount,
        unsigned int* searchResults) {

#define needleDescriptorIndex blockIdx.x
    assert(blockDim.x == 32);

    __shared__ SpinImage::gpu::FPFHHistogram33 referenceDescriptor;

    referenceDescriptor.contents[threadIdx.x] = needleDescriptors[needleDescriptorIndex].contents[threadIdx.x];

    // Descriptor has length 33, so with a single warp (32 threads) we need to manually copy the last one with thread 0
    if(threadIdx.x == 0) {
        referenceDescriptor.contents[32] = needleDescriptors[needleDescriptorIndex].contents[32];
    }

    __syncthreads();

    float referenceDescriptorAverage = computeDescriptorAverage(&referenceDescriptor);

    if(referenceDescriptorAverage == 0) {
        if(threadIdx.x == 0) {
            atomicAdd(&searchResults[needleDescriptorIndex], 512);
        }
        return;
    }

    float referenceCorrelation = computeFPFHDescriptorSimilarity(
            &referenceDescriptor,
            referenceDescriptorAverage,
            &haystackDescriptors[needleDescriptorIndex]);

    // No image pair can have a better correlation than 1, so we can just stop the search right here
    if(referenceCorrelation == 1) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackDescriptorCount; haystackImageIndex++) {
        if(needleDescriptorIndex == haystackImageIndex) {
            continue;
        }

        if(blockIdx.x == 0) {
            if(threadIdx.x == 0) {
                printf("%i: ", haystackImageIndex);
            }
            printf("%f, ", haystackDescriptors[haystackImageIndex].contents[threadIdx.x]);
            if(threadIdx.x == 0) {
                printf("%f\n", haystackDescriptors[haystackImageIndex].contents[32]);
            }
        }

        float correlation = computeFPFHDescriptorSimilarity(
                &referenceDescriptor,
                referenceDescriptorAverage,
                &haystackDescriptors[haystackImageIndex]);

        // We've found a result that's better than the reference one. That means this search result would end up
        // above ours in the search result list. We therefore move our search result down by 1.
        if(correlation > referenceCorrelation) {
            searchResultRank++;
        }
    }

    if(threadIdx.x == 0) {
        atomicAdd(&searchResults[needleDescriptorIndex], searchResultRank);
    }
}


SpinImage::array<unsigned int> SpinImage::gpu::computeFPFHSearchResultRanks(
        SpinImage::gpu::FPFHHistograms device_needleDescriptors,
        size_t needleDescriptorCount,
        SpinImage::gpu::FPFHHistograms device_haystackDescriptors,
        size_t haystackDescriptorCount,
        SpinImage::debug::FPFHSearchRunInfo* runInfo) {

    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = needleDescriptorCount * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();

    computeFPFHSearchResultIndices<<<needleDescriptorCount, 32>>>(
         device_needleDescriptors.histograms,
         device_haystackDescriptors.histograms,
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
    checkCudaErrors(cudaFree(device_searchResults));

    std::chrono::milliseconds executionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - executionStart);

    if(runInfo != nullptr) {
        runInfo->searchExecutionTimeSeconds = double(searchDuration.count()) / 1000.0;
        runInfo->totalExecutionTimeSeconds = double(executionDuration.count()) / 1000.0;
    }

    return resultIndices;
}



