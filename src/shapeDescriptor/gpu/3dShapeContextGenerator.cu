#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_math.h>
#endif

#include <shapeDescriptor/shapeDescriptor.h>

#include <chrono>
#include <iostream>
#include <cmath>
#include <shapeDescriptor/descriptors/ShapeContextGenerator.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__device__ bool operator==(float3 &a, float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ bool operator==(const float3 &a, const float3 &b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

__device__ bool operator==(float2 &a, float2 &b) {
    return a.x == b.x && a.y == b.y;
}

__device__ bool operator==(const float2 &a, const float2 &b) {
    return a.x == b.x && a.y == b.y;
}



// Run once for every vertex index
__global__ void createDescriptors(
        ShapeDescriptor::OrientedPoint* device_spinImageOrigins,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors,
        ShapeDescriptor::gpu::array<unsigned int> pointDensityArray,
        size_t sampleCount,
        float minSupportRadius,
        float maxSupportRadius)
{
#define descriptorIndex blockIdx.x
    const size_t elementsPerShapeContextDescriptor =
            SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
            SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
            SHAPE_CONTEXT_LAYER_COUNT;
    const ShapeDescriptor::OrientedPoint spinOrigin = device_spinImageOrigins[descriptorIndex];

    const float3 vertex = spinOrigin.vertex;
    float3 normal = spinOrigin.normal;

    normal /= length(normal);

    __shared__ ShapeDescriptor::ShapeContextDescriptor localDescriptor;
    for(int i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        localDescriptor.contents[i] = 0;
    }

    __syncthreads();

    // First, we align the input vertex with the descriptor's coordinate system
    float3 arbitraryAxis = {0, 0, 1};
    if(normal == arbitraryAxis || -normal == arbitraryAxis) {
        arbitraryAxis = {1, 0, 0};
    }

    float3 referenceXAxis = cross(arbitraryAxis, normal);
    float3 referenceYAxis = cross(referenceXAxis, normal);

    assert(length(referenceXAxis) != 0);
    assert(length(referenceYAxis) != 0);

    referenceXAxis /= length(referenceXAxis);
    referenceYAxis /= length(referenceYAxis);

    for (unsigned int sampleIndex = threadIdx.x; sampleIndex < sampleCount; sampleIndex += blockDim.x) {
        // 0. Fetch sample vertex

        const float3 samplePoint = pointCloud.vertices.at(sampleIndex);

        // 1. Compute bin indices

        const float3 translated = samplePoint - vertex;

        // Only include vertices which are within the support radius
        float distanceToVertex = length(translated);
        if (distanceToVertex < minSupportRadius || distanceToVertex > maxSupportRadius) {
            continue;
        }

        // Transforming descriptor coordinate system to the origin
        // In the new system, 'z' is 'up'
        const float3 relativeSamplePoint = {
                referenceXAxis.x * translated.x + referenceXAxis.y * translated.y + referenceXAxis.z * translated.z,
                referenceYAxis.x * translated.x + referenceYAxis.y * translated.y + referenceYAxis.z * translated.z,
                normal.x * translated.x + normal.y * translated.y + normal.z * translated.z,
        };

        float2 horizontalDirection = {relativeSamplePoint.x, relativeSamplePoint.y};
        float2 verticalDirection = {length(horizontalDirection), relativeSamplePoint.z};

        if (horizontalDirection == make_float2(0, 0)) {
            // special case, will result in an angle of 0
            horizontalDirection = {1, 0};

            // Vertical direction is only 0 if all components are 0
            // Should theoretically never occur, but let's handle it just in case
            if (verticalDirection.y == 0) {
                verticalDirection = {1, 0};
            }
        }

        // normalise direction vectors
        horizontalDirection /= length(horizontalDirection);
        verticalDirection /= length(verticalDirection);

        float horizontalAngle = ShapeDescriptor::internal::absoluteAngle(horizontalDirection.y, horizontalDirection.x);
        short horizontalIndex =
                unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                         float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
                % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;

        float verticalAngle = std::fmod(ShapeDescriptor::internal::absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                                        2.0f * float(M_PI));
        short verticalIndex =
                unsigned((verticalAngle / M_PI) *
                         float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
                % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

        float sampleDistance = length(relativeSamplePoint);
        short layerIndex = 0;

        // Recomputing logarithms is still preferable over doing memory transactions for each of them
        for (; layerIndex < SHAPE_CONTEXT_LAYER_COUNT; layerIndex++) {
            float nextSliceEnd = ShapeDescriptor::internal::computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
            if (sampleDistance <= nextSliceEnd) {
                break;
            }
        }

        // Rounding errors can cause it to exceed its allowed bounds in specific cases
        // Of course, on the off chance something is wrong after all,
        // the assertions further down should trip. So we only handle the single
        // edge case where layerIndex went one over.
        if(layerIndex == SHAPE_CONTEXT_LAYER_COUNT) {
            layerIndex--;
        }

        short3 binIndex = {horizontalIndex, verticalIndex, layerIndex};
        assert(binIndex.x >= 0);
        assert(binIndex.y >= 0);
        assert(binIndex.z >= 0);
        assert(binIndex.x < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT);
        assert(binIndex.y < SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
        assert(binIndex.z < SHAPE_CONTEXT_LAYER_COUNT);

        // 2. Compute sample weight
        float binVolume = ShapeDescriptor::internal::computeBinVolume(binIndex.y, binIndex.z, minSupportRadius, maxSupportRadius);

        // Volume can't be 0, and should be less than the volume of the support volume
        assert(binVolume > 0);
        assert(binVolume < (4.0f / 3.0f) * M_PI * maxSupportRadius * maxSupportRadius * maxSupportRadius);

        float sampleWeight = 1.0f / (pointDensityArray.content[sampleIndex] * std::cbrt(binVolume));

        // 3. Increment appropriate bin
        unsigned int index =
                binIndex.x * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT +
                binIndex.y * SHAPE_CONTEXT_LAYER_COUNT +
                binIndex.z;
        assert(index < elementsPerShapeContextDescriptor);
        assert(!isnan(sampleWeight));
        atomicAdd(&localDescriptor.contents[index], sampleWeight);
    }

    __syncthreads();

    // Copy final descriptor into memory

    for(size_t i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        descriptors.content[descriptorIndex].contents[i] = localDescriptor.contents[i];
    }

}
#endif

#define COMMA ,
ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> ShapeDescriptor::generate3DSCDescriptors(
        ShapeDescriptor::gpu::PointCloud device_pointCloud,
        ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_spinImageOrigins,
        float pointDensityRadius,
        float minSupportRadius,
        float maxSupportRadius,
        ShapeDescriptor::SCExecutionTimes* executionTimes) {
    CUDA_REGION(
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t descriptorCount = device_spinImageOrigins.length;
    size_t descriptorBufferSize = sizeof(ShapeDescriptor::ShapeContextDescriptor) * descriptorCount;

    ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_descriptors;
    device_descriptors.length = 0;
    device_descriptors.content = nullptr;

    // -- Initialisation --
    auto initialisationStart = std::chrono::steady_clock::now();

    checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));
    device_descriptors.length = descriptorCount;

    cudaMemset(device_descriptors.content, 0, descriptorBufferSize);
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);

    // -- Point Count Computation --
    auto pointCountingStart = std::chrono::steady_clock::now();

    ShapeDescriptor::gpu::array<unsigned int> device_pointCountArray =
            ShapeDescriptor::computePointDensities(pointDensityRadius, device_pointCloud);

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- Spin Image Generation --
    auto generationStart = std::chrono::steady_clock::now();

    createDescriptors <<<descriptorCount COMMA 416>>>(
        device_spinImageOrigins.content,
        device_pointCloud,
        device_descriptors,
        device_pointCountArray,
        device_pointCloud.pointCount,
        minSupportRadius,
        maxSupportRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --
    checkCudaErrors(cudaFree(device_pointCountArray.content));

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(executionTimes != nullptr) {
        executionTimes->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        executionTimes->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        executionTimes->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
        executionTimes->pointCountingTimeSeconds = double(pointCountingDuration.count()) / 1000.0;
    }

    return device_descriptors;
    )
}


