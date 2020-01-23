#include <spinImage/gpu/types/SampleBounds.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <spinImage/utilities/setValue.cuh>
#include <chrono>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <nvidia/helper_math.h>
#include "3dShapeContextGenerator.cuh"

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

const size_t elementsPerShapeContextDescriptor =
        SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT *
        SHAPE_CONTEXT_VERTICAL_SLICE_COUNT *
        SHAPE_CONTEXT_LAYER_COUNT;

__device__ __inline__ SpinImage::SampleBounds calculateSampleBounds(const SpinImage::array<float> &areaArray, int triangleIndex, int sampleCount) {
    SpinImage::SampleBounds sampleBounds;
    float maxArea = areaArray.content[areaArray.length - 1];
    float areaStepSize = maxArea / (float)sampleCount;

    if (triangleIndex == 0)
    {
        sampleBounds.areaStart = 0;
        sampleBounds.areaEnd = areaArray.content[0];
    }
    else
    {
        sampleBounds.areaStart = areaArray.content[triangleIndex - 1];
        sampleBounds.areaEnd = areaArray.content[triangleIndex];
    }

    size_t firstIndexInRange = (size_t) (sampleBounds.areaStart / areaStepSize) + 1;
    size_t lastIndexInRange = (size_t) (sampleBounds.areaEnd / areaStepSize);

    sampleBounds.sampleCount = lastIndexInRange - firstIndexInRange + 1; // Offset is needed to ensure bounds are correct
    sampleBounds.sampleStartIndex = firstIndexInRange - 1;

    return sampleBounds;
}

__device__ __inline__ float computeLayerDistance(float minSupportRadius, float maxSupportRadius, short layerIndex) {
    return std::exp(
            std::log(minSupportRadius)
            + (float(layerIndex) / float(SHAPE_CONTEXT_LAYER_COUNT))
            * std::log(float(maxSupportRadius) / float(minSupportRadius))
        );
}

__device__ __inline__ float computeWedgeSegmentVolume(short verticalBinIndex, float radius) {
    const float verticalAngleStep = 1.0f / float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT);
    float binStartAngle = float(verticalBinIndex) * verticalAngleStep;
    float binEndAngle = float(verticalBinIndex + 1) * verticalAngleStep;

    float scaleFraction = (2.0f * float(M_PI) * radius * radius * radius)
                        / (3.0f * float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT));
    return scaleFraction * (std::cos(binStartAngle) - std::cos(binEndAngle));
}

__device__ __inline__ float computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius) {
    // The wedge segment computation goes all the way from the center to the edge of the sphere
    // Since we also have a minimum support radius, we need to cut out the volume of the centre part
    float binEndRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
    float binStartRadius = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex);

    float largeSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binEndRadius);
    float smallSupportRadiusVolume = computeWedgeSegmentVolume(verticalBinIndex, binStartRadius);

    return largeSupportRadiusVolume - smallSupportRadiusVolume;
}

__inline__ __device__ float absoluteAngle(float y, float x) {
    float absoluteAngle = std::atan2(y, x);
    return absoluteAngle < 0 ? absoluteAngle + (2.0f * float(M_PI)) : absoluteAngle;
}

__inline__ __device__ unsigned int warpAllReduceSum(unsigned int val) {
    const int warpSize = 32;
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xFFFFFFFF, val, mask);
    return val;
}

__global__ void computePointCounts(
        SpinImage::array<unsigned int> pointDensityArray,
        SpinImage::gpu::PointCloud pointCloud,
        float countRadius) {

    assert(blockDim.x == 32);

    unsigned int pointIndex = blockIdx.x;
    unsigned int threadPointCount = 0;
    float3 referencePoint = pointCloud.vertices.at(pointIndex);

    for (unsigned int samplePointIndex = threadIdx.x; samplePointIndex < pointCloud.vertices.length; samplePointIndex += blockDim.x)
    {
        float3 samplePoint = pointCloud.vertices.at(samplePointIndex);
        float3 delta = samplePoint - referencePoint;
        float distanceToPoint = length(delta);
        if(distanceToPoint <= countRadius) {
            threadPointCount++;
        }
    }

    unsigned int totalPointCount = warpAllReduceSum(threadPointCount);
    if(threadIdx.x == 0) {
        pointDensityArray.content[pointIndex] = totalPointCount;
    }
}

// Run once for every vertex index
__global__ void createDescriptors(
        SpinImage::gpu::Mesh mesh,
        SpinImage::gpu::DeviceOrientedPoint* device_spinImageOrigins,
        SpinImage::gpu::PointCloud pointCloud,
        SpinImage::array<shapeContextBinType> descriptors,
        SpinImage::array<float> areaArray,
        SpinImage::array<unsigned int> pointDensityArray,
        size_t sampleCount,
        float minSupportRadius,
        float maxSupportRadius)
{
#define descriptorIndex blockIdx.x

    const SpinImage::gpu::DeviceOrientedPoint spinOrigin = device_spinImageOrigins[descriptorIndex];

    const float3 vertex = spinOrigin.vertex;
    const float3 normal = spinOrigin.normal;

    __shared__ shapeContextBinType localDescriptor[elementsPerShapeContextDescriptor];
    for(int i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        localDescriptor[i] = 0;
    }

    __syncthreads();

    // First, we align the input vertex with the descriptor's coordinate system
    float3 arbitraryAxis = {0, 0, 1};
    if(normal == arbitraryAxis) {
        arbitraryAxis = {1, 0, 0};
    }

    const float3 referenceXAxis = cross(arbitraryAxis, normal);
    const float3 referenceYAxis = cross(referenceXAxis, normal);

    for (int triangleIndex = threadIdx.x; triangleIndex < mesh.vertexCount / 3; triangleIndex += blockDim.x)
    {
        SpinImage::SampleBounds bounds = calculateSampleBounds(areaArray, triangleIndex, sampleCount);

        for(unsigned int sample = 0; sample < bounds.sampleCount; sample++)
        {
            // 0. Fetch sample vertex
            size_t sampleIndex = bounds.sampleStartIndex + sample;

            if(sampleIndex >= sampleCount) {
                printf("Sample %i/%i/%i was skipped.\n", sampleIndex, bounds.sampleCount, sampleCount);
                continue;
            }

            const float3 samplePoint = pointCloud.vertices.at(sampleIndex);

            // 1. Compute bin indices

            const float3 translated = samplePoint - vertex;

            // Only include vertices which are within the support radius
            float distanceToVertex = length(translated);
            if(distanceToVertex < minSupportRadius || distanceToVertex > maxSupportRadius) {
                continue;
            }

            // Transforming descriptor coordinate system to the origin
            // In the new system, 'z' is 'up'
            const float3 relativeSamplePoint = {
                referenceXAxis.x * translated.x + referenceXAxis.y * translated.y + referenceXAxis.z * translated.z,
                referenceYAxis.x * translated.x + referenceYAxis.y * translated.y + referenceYAxis.z * translated.z,
                normal.x         * translated.x + normal.y         * translated.y + normal.z         * translated.z,
            };

            float2 horizontalDirection = {relativeSamplePoint.x, relativeSamplePoint.y};
            float2 verticalDirection = {length(horizontalDirection), relativeSamplePoint.z};

            if(horizontalDirection == make_float2(0, 0)) {
                // special case, will result in an angle of 0
                horizontalDirection = {1, 0};

                // Vertical direction is only 0 if all components are 0
                // Should theoretically never occur, but let's handle it just in case
                if(verticalDirection.y == 0) {
                    verticalDirection = {1, 0};
                }
            }

            // normalise direction vector
            horizontalDirection /= length(horizontalDirection);
            verticalDirection /= length(verticalDirection);

            float horizontalAngle = absoluteAngle(horizontalDirection.y, horizontalDirection.x);
            short horizontalIndex =
                    unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                    float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
                    % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;


            short verticalIndex =
                    unsigned((verticalAngle / M_PI) *
                    float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
                    % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

            float sampleDistance = length(relativeSamplePoint);
            short layerIndex = 0;

            // Recomputing logarithms is still preferable over doing memory transactions for each of them
            for(; layerIndex <= SHAPE_CONTEXT_LAYER_COUNT; layerIndex++) {

                    break;
                }
            }

            short3 binIndex = {horizontalIndex, verticalIndex, layerIndex};

            // 2. Compute sample weight
            float binVolume = computeBinVolume(binIndex.y, binIndex.z, minSupportRadius, maxSupportRadius);
            float sampleWeight = 1.0f / pointDensityArray.content[sampleIndex] * std::cbrt(binVolume);

            // 3. Increment appropriate bin
            unsigned int index =
                    binIndex.x * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT +
                    binIndex.y * SHAPE_CONTEXT_LAYER_COUNT +
                    binIndex.z;
            atomicAdd(&localDescriptor[index], sampleWeight);
        }
    }

    __syncthreads();

    // Copy final descriptor into memory

    size_t descriptorBaseIndex = size_t(descriptorIndex) * elementsPerShapeContextDescriptor;
    for(size_t i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        descriptors.content[descriptorBaseIndex + i] = localDescriptor[i];
    }

}

SpinImage::array<shapeContextBinType> SpinImage::gpu::generate3DSCDescriptors(
        Mesh device_mesh,
        array<DeviceOrientedPoint> device_spinImageOrigins,
        float pointDensityRadius,
        float minSupportRadius,
        float maxSupportRadius,
        size_t sampleCount,
        size_t randomSamplingSeed,
        SpinImage::debug::SCRunInfo* runInfo) {
    auto totalExecutionTimeStart = std::chrono::steady_clock::now();

    size_t descriptorCount = device_spinImageOrigins.length;

    size_t descriptorBufferLength = descriptorCount * elementsPerShapeContextDescriptor;
    size_t descriptorBufferSize = sizeof(shapeContextBinType) * descriptorBufferLength;

    array<shapeContextBinType> device_descriptors = {0, nullptr};

    // -- Initialisation --
    auto initialisationStart = std::chrono::steady_clock::now();

    checkCudaErrors(cudaMalloc(&device_descriptors.content, descriptorBufferSize));

    device_descriptors.length = descriptorCount;

    CudaLaunchDimensions valueSetSettings = calculateCudaLaunchDimensions(descriptorBufferLength);
    setValue <shapeContextBinType><<<valueSetSettings.blocksPerGrid, valueSetSettings.threadsPerBlock >>> (device_descriptors.content, descriptorBufferLength, 0);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds initialisationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - initialisationStart);

    // -- Mesh Sampling --
    auto meshSamplingStart = std::chrono::steady_clock::now();

    SpinImage::internal::MeshSamplingBuffers sampleBuffers;
    PointCloud device_pointCloud = SpinImage::utilities::sampleMesh(device_mesh, sampleCount, randomSamplingSeed, &sampleBuffers);
    array<float> device_cumulativeAreaArray = sampleBuffers.cumulativeAreaArray;

    std::chrono::milliseconds meshSamplingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - meshSamplingStart);

    // -- Point Count Computation --
    auto pointCountingStart = std::chrono::steady_clock::now();
    SpinImage::array<unsigned int> device_pointCountArray = {sampleCount, nullptr};
    checkCudaErrors(cudaMalloc(&device_pointCountArray.content, sampleCount * sizeof(unsigned int)));
    computePointCounts<<<sampleCount, 32>>>(device_pointCountArray, device_pointCloud, pointDensityRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds pointCountingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - pointCountingStart);

    // -- Spin Image Generation --
    auto generationStart = std::chrono::steady_clock::now();

    createDescriptors <<<descriptorCount, 416>>>(
            device_mesh,
                    device_spinImageOrigins.content,
                    device_pointCloud,
                    device_descriptors,
                    device_cumulativeAreaArray,
                    device_pointCountArray,
                    sampleCount,
                    minSupportRadius,
                    maxSupportRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::chrono::milliseconds generationDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generationStart);

    // -- Cleanup --

    checkCudaErrors(cudaFree(device_cumulativeAreaArray.content));
    checkCudaErrors(cudaFree(device_pointCountArray.content));
    device_pointCloud.vertices.free();
    device_pointCloud.normals.free();

    std::chrono::milliseconds totalExecutionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - totalExecutionTimeStart);

    if(runInfo != nullptr) {
        runInfo->totalExecutionTimeSeconds = double(totalExecutionDuration.count()) / 1000.0;
        runInfo->initialisationTimeSeconds = double(initialisationDuration.count()) / 1000.0;
        runInfo->meshSamplingTimeSeconds = double(meshSamplingDuration.count()) / 1000.0;
        runInfo->generationTimeSeconds = double(generationDuration.count()) / 1000.0;
        runInfo->pointCountingTimeSeconds = double(pointCountingDuration.count()) / 1000.0;
    }

    return device_descriptors;
}
