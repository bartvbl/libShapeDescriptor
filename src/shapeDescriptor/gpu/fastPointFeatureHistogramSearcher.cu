#include <iostream>
#include <chrono>
#include <cassert>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#endif

#include <shapeDescriptor/shapeDescriptor.h>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float computeDescriptorAverage(ShapeDescriptor::FPFHDescriptor &descriptor) {
    float threadSum = 0;
    for(unsigned int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
        threadSum += descriptor.contents[i];
    }
    float totalSum = warpAllReduceSum(threadSum);
    return totalSum / float(3 * FPFH_BINS_PER_FEATURE);
}

__device__ float computeFPFHDescriptorSimilarity(
        ShapeDescriptor::FPFHDescriptor &needleDescriptor,
        float needleDescriptorAverage,
        ShapeDescriptor::FPFHDescriptor &haystackDescriptor) {

    float haystackDescriptorAverage = computeDescriptorAverage(haystackDescriptor);

    float threadSquaredSumX = 0;
    float threadSquaredSumY = 0;
    float threadMultiplicativeSum = 0;

    for(int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += warpSize) {
        float needleDescriptorValue = needleDescriptor.contents[i];
        float haystackDescriptorValue = haystackDescriptor.contents[i];

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
        ShapeDescriptor::FPFHDescriptor* needleDescriptors,
        ShapeDescriptor::FPFHDescriptor* haystackDescriptors,
        size_t haystackDescriptorCount,
        unsigned int* searchResults) {

#define needleDescriptorIndex blockIdx.x
    assert(blockDim.x == 32);

    __shared__ ShapeDescriptor::FPFHDescriptor referenceDescriptor;
    __shared__ ShapeDescriptor::FPFHDescriptor haystackDescriptor;

    for(unsigned int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
        referenceDescriptor.contents[i] = needleDescriptors[needleDescriptorIndex].contents[i];
        haystackDescriptor.contents[i] = haystackDescriptors[needleDescriptorIndex].contents[i];
    }

    __syncthreads();

    float referenceDescriptorAverage = computeDescriptorAverage(referenceDescriptor);

    if(referenceDescriptorAverage == 0) {
        if(threadIdx.x == 0) {
            printf("WARNING: descriptor %i consists entirely of zeroes!\n", needleDescriptorIndex);
            // Effectively remove the descriptor from the list of search results
            atomicAdd(&searchResults[needleDescriptorIndex], 1 << 30);
        }
        return;
    }

    float referenceCorrelation = computeFPFHDescriptorSimilarity(
            referenceDescriptor,
            referenceDescriptorAverage,
            haystackDescriptor);

    // No image pair can have a better correlation than 1, so we can just stop the search right here
    if(referenceCorrelation == 1) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackImageIndex = 0; haystackImageIndex < haystackDescriptorCount; haystackImageIndex++) {
        if(needleDescriptorIndex == haystackImageIndex) {
            continue;
        }

        for(unsigned int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
            haystackDescriptor.contents[i] = haystackDescriptors[haystackImageIndex].contents[i];
        }

        float correlation = computeFPFHDescriptorSimilarity(
                referenceDescriptor,
                referenceDescriptorAverage,
                haystackDescriptor);

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
#endif

ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::computeFPFHSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_needleDescriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_haystackDescriptors,
        ShapeDescriptor::FPFHSearchExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    auto searchStart = std::chrono::steady_clock::now();


    computeFPFHSearchResultIndices<<<device_needleDescriptors.length, 32>>>(
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
    checkCudaErrors(cudaFree(device_searchResults));

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
__global__ void computeElementWiseFPFHEuclideanDistances(
        ShapeDescriptor::FPFHDescriptor* descriptors,
        ShapeDescriptor::FPFHDescriptor* correspondingDescriptors,
        float* distances) {
    const size_t descriptorIndex = blockIdx.x;

    static_assert(spinImageWidthPixels % 32 == 0, "This kernel assumes the image is a multiple of the warp size wide");

    size_t needleImageIndex = blockIdx.x;

    float threadSquaredSum = 0;

    for(unsigned int i = threadIdx.x; i < 3 * FPFH_BINS_PER_FEATURE; i += blockDim.x) {
        float descriptorPixelValue = descriptors[needleImageIndex].contents[i];
        float correspondingPixelValue = correspondingDescriptors[needleImageIndex].contents[i];
        float pixelDelta = descriptorPixelValue - correspondingPixelValue;
        threadSquaredSum += pixelDelta * pixelDelta;
    }

    float totalSquaredSum = warpAllReduceSum(threadSquaredSum);

    if(threadIdx.x == 0) {
        distances[descriptorIndex] = sqrt(totalSquaredSum);
    }
}
#endif

ShapeDescriptor::cpu::array<float> ShapeDescriptor::computeFPFHElementWiseEuclideanDistances(
        ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_descriptors,
        ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_correspondingDescriptors) {
    ShapeDescriptor::gpu::array<float> device_distances(device_descriptors.length);

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    computeElementWiseFPFHEuclideanDistances<<<device_descriptors.length, 32>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            device_distances.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<float> distances = device_distances.copyToCPU();

    ShapeDescriptor::free(device_distances);

    return distances;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}