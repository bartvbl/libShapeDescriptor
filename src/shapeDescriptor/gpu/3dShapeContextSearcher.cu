#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#endif

#include <chrono>
#include <cfloat>
#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
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
        ShapeDescriptor::ShapeContextDescriptor &needleDescriptor,
        ShapeDescriptor::ShapeContextDescriptor &haystackDescriptor,
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
        ShapeDescriptor::ShapeContextDescriptor* needleDescriptors,
        ShapeDescriptor::ShapeContextDescriptor* haystackDescriptors,
        size_t haystackDescriptorCount,
        float haystackScaleFactor,
        unsigned int* searchResults) {
#define needleDescriptorIndex blockIdx.x

    // Since memory is reused a lot, we cache both the needle and haystack image in shared memory
    // Combined this is is approximately (at default settings) the size of a spin or RICI image

    __shared__ ShapeDescriptor::ShapeContextDescriptor referenceDescriptor;
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        referenceDescriptor.contents[index] = needleDescriptors[needleDescriptorIndex].contents[index];
    }

    __shared__ ShapeDescriptor::ShapeContextDescriptor haystackDescriptor;
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
#endif


ShapeDescriptor::cpu::array<unsigned int> ShapeDescriptor::compute3DSCSearchResultRanks(
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_needleDescriptors,
        size_t needleDescriptorSampleCount,
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_haystackDescriptors,
        size_t haystackDescriptorSampleCount,
        ShapeDescriptor::SCSearchExecutionTimes* executionTimes) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    static_assert(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT <= 32, "Exceeding this number of slices causes an overflow in the amount of shared memory needed by the kernel");

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
__global__ void computeElementWiseDistances3DSC(
        ShapeDescriptor::ShapeContextDescriptor* descriptors,
        ShapeDescriptor::ShapeContextDescriptor* correspondingDescriptors,
        size_t descriptorCount,
        float haystackScaleFactor,
        float* distances) {
#define needleDescriptorIndex blockIdx.x

    // Since memory is reused a lot, we cache both the needle and haystack image in shared memory
    // Combined this is is approximately (at default settings) the size of a spin or RICI image

    __shared__ ShapeDescriptor::ShapeContextDescriptor descriptor;
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        descriptor.contents[index] = descriptors[needleDescriptorIndex].contents[index];
    }

    __shared__ ShapeDescriptor::ShapeContextDescriptor correspondingDescriptor;
    for(unsigned int index = blockDim.x * threadIdx.y + threadIdx.x; index < elementsPerShapeContextDescriptor; index += blockDim.x * blockDim.y) {
        correspondingDescriptor.contents[index] =
                correspondingDescriptors[needleDescriptorIndex].contents[index] * (1.0f/haystackScaleFactor);
    }

    __shared__ float squaredSums[SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT];

    __syncthreads();

    float distance = compute3DSCPairDistanceGPU(
            descriptor,
            correspondingDescriptor,
            squaredSums);

    if(threadIdx.x == 0) {
        distances[needleDescriptorIndex] = distance;
    }
}
#endif


ShapeDescriptor::cpu::array<float> ShapeDescriptor::compute3DSCElementWiseSquaredDistances(
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_descriptors,
        size_t descriptorSampleCount,
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_correspondingDescriptors,
        size_t correspondingDescriptorsSampleCount) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    static_assert(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT <= 32, "Exceeding this number of slices causes an overflow in the amount of shared memory needed by the kernel");
    assert(device_descriptors.length == device_correspondingDescriptors.length);

    ShapeDescriptor::gpu::array<float> device_results(device_descriptors.length);

    float haystackScaleFactor = float(double(descriptorSampleCount) / double(correspondingDescriptorsSampleCount));

    dim3 blockDimensions = {
            32, SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT, 1
    };
    computeElementWiseDistances3DSC<<<device_descriptors.length, blockDimensions>>>(
            device_descriptors.content,
            device_correspondingDescriptors.content,
            device_descriptors.length,
            haystackScaleFactor,
            device_results.content);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    ShapeDescriptor::cpu::array<float> distances = device_results.copyToCPU();
    ShapeDescriptor::free(device_results);

    return distances;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}