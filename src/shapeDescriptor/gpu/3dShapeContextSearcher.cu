#include <algorithm>
#include <chrono>
#include "3dShapeContextSearcher.cuh"
#include <nvidia/helper_cuda.h>
#include <cfloat>
#include <host_defines.h>
#include <iostream>
#include <vector_types.h>
#include <shapeDescriptor/gpu/types/methods/3DSCDescriptor.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;


__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__inline__ __device__ float warpAllReduceMin(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val = min(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    return val;
}

__device__ float compute3DSCPairDistanceGPU(
        SpinImage::gpu::ShapeContextDescriptor &needleDescriptor,
        SpinImage::gpu::ShapeContextDescriptor &haystackDescriptor,
        float* sharedSquaredSums) {

#define sliceOffset threadIdx.y
    float threadSquaredDistance = 0;
    for(short binIndex = threadIdx.x; binIndex < elementsPerShapeContextDescriptor; binIndex += blockDim.x) {
        float needleBinValue = needleDescriptor.contents[binIndex];
        short haystackBinIndex =
            (binIndex + (sliceOffset * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT));
        // Simple modulo that I think is less expensive
        if(haystackBinIndex >= elementsPerShapeContextDescriptor) {
            haystackBinIndex -= elementsPerShapeContextDescriptor;
        }
        float haystackBinValue = haystackDescriptor.contents[haystackBinIndex];
        float binDelta = needleBinValue - haystackBinValue;
        threadSquaredDistance += binDelta * binDelta;
    }

    float combinedSquaredDistance = warpAllReduceSum(threadSquaredDistance);

    if(threadIdx.x == 0) {
        sharedSquaredSums[sliceOffset] = combinedSquaredDistance;
    }

    __syncthreads();

    // An entire warp must participate in the reduction, so we give the excess threads
    // the highest possible value so that any other value will be lower
    float threadValue = threadIdx.x < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT ?
            sharedSquaredSums[threadIdx.x] : FLT_MAX;
    float lowestDistance = std::sqrt(warpAllReduceMin(threadValue));

    // Some threads will race ahead to the next image pair. Need to avoid that.
    __syncthreads();

    return lowestDistance;
}

__global__ void computeShapeContextSearchResultIndices(
        SpinImage::gpu::ShapeContextDescriptor* needleDescriptors,
        SpinImage::gpu::ShapeContextDescriptor* haystackDescriptors,
        size_t haystackDescriptorCount,
        float haystackScaleFactor,
        unsigned int* searchResults) {
#define needleDescriptorIndex blockIdx.x

    // Since memory is reused a lot, we cache both the needle and haystack image in shared memory
    // Combined this is is approximately (at default settings) the size of a spin or RICI image

    __shared__ SpinImage::gpu::ShapeContextDescriptor referenceDescriptor;
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        referenceDescriptor.contents[index] = needleDescriptors[needleDescriptorIndex].contents[index];
    }

    __shared__ SpinImage::gpu::ShapeContextDescriptor haystackDescriptor;
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        haystackDescriptor.contents[index] =
                haystackDescriptors[needleDescriptorIndex].contents[index] * (1.0f/haystackScaleFactor);
    }

    __shared__ float squaredSums[SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT];

    __syncthreads();

    float referenceDistance = compute3DSCPairDistanceGPU(
            referenceDescriptor,
            haystackDescriptor,
            squaredSums);

    // No image pair can have a better distance than 0, so we can just stop the search right here
    if(referenceDistance == 0) {
        return;
    }

    unsigned int searchResultRank = 0;

    for(size_t haystackDescriptorIndex = 0; haystackDescriptorIndex < haystackDescriptorCount; haystackDescriptorIndex++) {
        if(needleDescriptorIndex == haystackDescriptorIndex) {
            continue;
        }

        for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
            haystackDescriptor.contents[index] =
                    haystackDescriptors[haystackDescriptorIndex].contents[index] * (1.0f/haystackScaleFactor);
        }

        __syncthreads();

        float distance = compute3DSCPairDistanceGPU(
                referenceDescriptor,
                haystackDescriptor,
                squaredSums);

        // We've found a result that's better than the reference one. That means this search result would end up
        // above ours in the search result list. We therefore move our search result down by 1.
        if(distance < referenceDistance) {
            searchResultRank++;
        }
    }

    if(threadIdx.x == 0) {
        searchResults[needleDescriptorIndex] = searchResultRank;
    }
}



SpinImage::cpu::array<unsigned int> SpinImage::gpu::compute3DSCSearchResultRanks(
        SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> device_needleDescriptors,
        size_t needleDescriptorSampleCount,
        SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> device_haystackDescriptors,
        size_t haystackDescriptorSampleCount,
        SpinImage::debug::SCSearchExecutionTimes* executionTimes) {
    static_assert(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT <= 32);

    auto executionStart = std::chrono::steady_clock::now();

    size_t searchResultBufferSize = device_needleDescriptors.length * sizeof(unsigned int);
    unsigned int* device_searchResults;
    checkCudaErrors(cudaMalloc(&device_searchResults, searchResultBufferSize));
    checkCudaErrors(cudaMemset(device_searchResults, 0, searchResultBufferSize));

    float haystackScaleFactor = float(double(needleDescriptorSampleCount) / double(haystackDescriptorSampleCount));
    std::cout << "\t\tHaystack scale factor: " << haystackScaleFactor << std::endl;

    auto searchStart = std::chrono::steady_clock::now();

    dim3 blockDimensions = {
        32, SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT, 1
    };
    computeShapeContextSearchResultIndices<<<device_needleDescriptors.length, blockDimensions>>>(
        device_needleDescriptors.content,
        device_haystackDescriptors.content,
        device_haystackDescriptors.length,
        haystackScaleFactor,
        device_searchResults);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds searchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - searchStart);

    SpinImage::cpu::array<unsigned int> resultIndices;
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