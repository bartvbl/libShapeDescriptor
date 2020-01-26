#include <spinImage/gpu/types/SampleBounds.h>
#include <spinImage/gpu/types/PointCloud.h>
#include <spinImage/gpu/types/CudaLaunchDimensions.h>
#include <spinImage/utilities/meshSampler.cuh>
#include <spinImage/utilities/setValue.cuh>
#include <chrono>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <nvidia/helper_math.h>
#include <spinImage/gpu/types/BoundingBox.h>
#include <spinImage/utilities/pointCloudUtils.h>
#include <iostream>
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

std::ostream& operator << (std::ostream &o, const float3& p)
{
    return o << "(" << p.x << ", " << p.y << ", " << p.z << ")";
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
    // Avoiding zero divisions
    if(minSupportRadius == 0) {
        minSupportRadius = 0.000001f;
    }
    return std::exp(
            (std::log(minSupportRadius))
            + ((float(layerIndex) / float(SHAPE_CONTEXT_LAYER_COUNT))
            * std::log(float(maxSupportRadius) / float(minSupportRadius)))
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

__device__ float absoluteAngle(float y, float x) {
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
        SpinImage::gpu::BoundingBox boundingBox,
        unsigned int* indexTable,
        int3 binCounts,
        float binSize,
        float countRadius) {

    assert(blockDim.x == 32);

    unsigned int pointIndex = blockIdx.x;
    unsigned int threadPointCount = 0;
    float3 referencePoint = pointCloud.vertices.at(pointIndex);

    float3 referencePointBoundsMin = referencePoint - float3{countRadius, countRadius, countRadius} - boundingBox.min;
    float3 referencePointBoundsMax = referencePoint + float3{countRadius, countRadius, countRadius} - boundingBox.min;

    int3 minBinIndices = {
        min(max(int(referencePointBoundsMin.x / binSize), 0), binCounts.x-1),
        min(max(int(referencePointBoundsMin.y / binSize), 0), binCounts.y-1),
        min(max(int(referencePointBoundsMin.z / binSize), 0), binCounts.z-1)
    };

    int3 maxBinIndices = {
        min(max(int(referencePointBoundsMax.x / binSize) + 2, 0), binCounts.x-1),
        min(max(int(referencePointBoundsMax.y / binSize) + 2, 0), binCounts.y-1),
        min(max(int(referencePointBoundsMax.z / binSize) + 2, 0), binCounts.z-1)
    };

    assert(minBinIndices.x < binCounts.x);
    assert(minBinIndices.y < binCounts.y);
    assert(minBinIndices.z < binCounts.z);
    assert(maxBinIndices.x < binCounts.x);
    assert(maxBinIndices.y < binCounts.y);
    assert(maxBinIndices.z < binCounts.z);

    /*if(threadIdx.x == 0) {
        printf("(%i, %i, %i) -> (%i, %i, %i)\n", minBinIndices.x, minBinIndices.y, minBinIndices.z, maxBinIndices.x, maxBinIndices.y, maxBinIndices.z);
    }*/

    // TODO: Ensure vertex does not count itself
    // TODO: Correctly handle end corner bin such that its contents are included in point count

    for(unsigned int binZ = minBinIndices.z; binZ < maxBinIndices.z; binZ++) {
        for(unsigned int binY = minBinIndices.y; binY < maxBinIndices.y; binY++) {
            unsigned int startTableIndex = binZ * binCounts.x * binCounts.y + binY * binCounts.x + minBinIndices.x;
            unsigned int endTableIndex = binZ * binCounts.x * binCounts.y + binY * binCounts.x + maxBinIndices.x;

            unsigned int startVertexIndex = indexTable[startTableIndex];
            unsigned int endVertexIndex = 0;

            if(endVertexIndex < binCounts.x * binCounts.y * binCounts.z) {
                endVertexIndex = indexTable[endTableIndex];
            } else {
                endVertexIndex = pointCloud.vertices.length;
            }

            assert(startVertexIndex <= endVertexIndex);
            assert(startVertexIndex < pointCloud.vertices.length);
            assert(endVertexIndex < pointCloud.vertices.length);

            for (unsigned int samplePointIndex = startVertexIndex + threadIdx.x; samplePointIndex < endVertexIndex; samplePointIndex += blockDim.x)
            {
                float3 samplePoint = pointCloud.vertices.at(samplePointIndex);
                float3 delta = samplePoint - referencePoint;
                float distanceToPoint = length(delta);
                if(distanceToPoint <= countRadius) {
                    threadPointCount++;
                }
            }
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
    float3 normal = spinOrigin.normal;

    //assert(length(normal) != 0);

    normal /= length(normal);

    __shared__ shapeContextBinType localDescriptor[elementsPerShapeContextDescriptor];
    for(int i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        localDescriptor[i] = 0;
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

        // normalise direction vector
        horizontalDirection /= length(horizontalDirection);
        verticalDirection /= length(verticalDirection);

        float horizontalAngle = absoluteAngle(horizontalDirection.y, horizontalDirection.x);
        short horizontalIndex =
                unsigned((horizontalAngle / (2.0f * float(M_PI))) *
                         float(SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT))
                % SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;

        float verticalAngle = std::fmod(absoluteAngle(verticalDirection.y, verticalDirection.x) + (float(M_PI) / 2.0f),
                                        2.0f * float(M_PI));
        short verticalIndex =
                unsigned((verticalAngle / M_PI) *
                         float(SHAPE_CONTEXT_VERTICAL_SLICE_COUNT))
                % SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

        float sampleDistance = length(relativeSamplePoint);
        short layerIndex = 0;

        // Recomputing logarithms is still preferable over doing memory transactions for each of them
        for (; layerIndex < SHAPE_CONTEXT_LAYER_COUNT; layerIndex++) {
            float nextSliceEnd = computeLayerDistance(minSupportRadius, maxSupportRadius, layerIndex + 1);
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
        float binVolume = computeBinVolume(binIndex.y, binIndex.z, minSupportRadius, maxSupportRadius);

        // Volume can't be 0, and should be less than the volume of the support volume
        assert(binVolume > 0);
        assert(binVolume < (4.0f / 3.0f) * M_PI * maxSupportRadius * maxSupportRadius * maxSupportRadius);

        float sampleWeight = 1.0f / pointDensityArray.content[sampleIndex] * std::cbrt(binVolume);

        // 3. Increment appropriate bin
        unsigned int index =
                binIndex.x * SHAPE_CONTEXT_LAYER_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT +
                binIndex.y * SHAPE_CONTEXT_LAYER_COUNT +
                binIndex.z;
        assert(index < elementsPerShapeContextDescriptor);
        assert(!std::isnan(sampleWeight));
        atomicAdd(&localDescriptor[index], sampleWeight);

    }

    __syncthreads();

    // Copy final descriptor into memory

    size_t descriptorBaseIndex = size_t(descriptorIndex) * elementsPerShapeContextDescriptor;
    for(size_t i = threadIdx.x; i < elementsPerShapeContextDescriptor; i += blockDim.x) {
        descriptors.content[descriptorBaseIndex + i] = localDescriptor[i];
    }

}

__global__ void countBinContents(
    SpinImage::gpu::PointCloud pointCloud,
    unsigned int* indexTable,
    SpinImage::gpu::BoundingBox boundingBox,
    int3 binCounts,
    float binSize) {

    unsigned int vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= pointCloud.vertices.length) {
        return;
    }

    float3 vertex = pointCloud.vertices.at(vertexIndex);

    float3 relativeToBoundingBox = vertex - boundingBox.min;

    int3 binIndex = {
        min(max(int(relativeToBoundingBox.x / binSize), 0), binCounts.x-1),
        min(max(int(relativeToBoundingBox.y / binSize), 0), binCounts.y-1),
        min(max(int(relativeToBoundingBox.z / binSize), 0), binCounts.z-1)
    };

    unsigned int indexTableIndex = binIndex.z * binCounts.x * binCounts.y + binIndex.y * binCounts.x + binIndex.x;

    assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

    atomicAdd(&indexTable[indexTableIndex], 1);
}

__global__ void countCumulativeBinIndices(unsigned int* indexTable, int3 binCounts, unsigned int pointCloudSize) {
    unsigned int cumulativeIndex = 0;
    for(int z = 0; z < binCounts.z; z++) {
        for(int y = 0; y < binCounts.y; y++) {
            for(int x = 0; x < binCounts.x; x++) {
                unsigned int binIndex = z * binCounts.x * binCounts.y + y * binCounts.x + x;
                unsigned int binLength = indexTable[binIndex];
                indexTable[binIndex] = cumulativeIndex;
                cumulativeIndex += binLength;
            }
        }
    }
    assert(cumulativeIndex == pointCloudSize);
}

__global__ void rearrangePointCloud(
        SpinImage::gpu::PointCloud sourcePointCloud,
        SpinImage::gpu::PointCloud destinationPointCloud,
        SpinImage::gpu::BoundingBox boundingBox,
        unsigned int* nextIndexEntryTable,
        int3 binCounts,
        float binSize) {
    unsigned int vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= sourcePointCloud.vertices.length) {
        return;
    }

    float3 vertex = sourcePointCloud.vertices.at(vertexIndex);

    float3 relativeToBoundingBox = vertex - boundingBox.min;

    int3 binIndex = {
            min(max(int(relativeToBoundingBox.x / binSize), 0), binCounts.x-1),
            min(max(int(relativeToBoundingBox.y / binSize), 0), binCounts.y-1),
            min(max(int(relativeToBoundingBox.z / binSize), 0), binCounts.z-1)
    };

    unsigned int indexTableIndex = binIndex.z * binCounts.x * binCounts.y + binIndex.y * binCounts.x + binIndex.x;

    assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

    unsigned int targetIndex = atomicAdd(&nextIndexEntryTable[indexTableIndex], 1);

    destinationPointCloud.vertices.set(targetIndex, vertex);
}

SpinImage::array<unsigned int> computePointDensities(float pointDensityRadius, SpinImage::gpu::PointCloud device_pointCloud, size_t &sampleCount) {
    // 1. Compute bounding box
    SpinImage::gpu::BoundingBox boundingBox = SpinImage::utilities::computeBoundingBox(device_pointCloud);
    std::cout << "Min: " << boundingBox.min << std::endl;
    std::cout << "Max: " << boundingBox.max << std::endl;

    // 2. Allocate index array for boxes of radius x radius x radius
    float3 boundingBoxSize = boundingBox.max - boundingBox.min;
    float binSize = std::cbrt(boundingBoxSize.x * boundingBoxSize.y * boundingBoxSize.z) / 50.0f;
    std::cout << "Box size: " << binSize << std::endl;
    int3 binCounts = {int(boundingBoxSize.x / binSize) + 1,
                      int(boundingBoxSize.y / binSize) + 1,
                      int(boundingBoxSize.z / binSize) + 1};
    int totalBinCount = binCounts.x * binCounts.y * binCounts.z;
    std::cout << "Bin counts: " << binCounts.x << ", " << binCounts.y << ", " << binCounts.z << std::endl;
    unsigned int* device_indexTable;
    checkCudaErrors(cudaMalloc(&device_indexTable, totalBinCount * sizeof(unsigned int)));
    checkCudaErrors(cudaMemset(device_indexTable, 0, totalBinCount * sizeof(unsigned int)));

    // 3. Counting occurrences for each box
    countBinContents<<<(device_pointCloud.vertices.length / 256) + 1, 256>>>(
            device_pointCloud, device_indexTable, boundingBox, binCounts, binSize);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 4. Compute cumulative indices
    // Single threaded, because there aren't all that many bins, and you don't win much by parallelising it anyway
    countCumulativeBinIndices<<<1, 1>>>(device_indexTable, binCounts, device_pointCloud.vertices.length);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // 5. Allocate temporary point cloud (vertices only)
    SpinImage::gpu::PointCloud device_tempPointCloud(device_pointCloud.vertices.length);

    // 6. Copy over contents of point cloud
    checkCudaErrors(cudaMemcpy(device_tempPointCloud.vertices.array, device_pointCloud.vertices.array,
        device_pointCloud.vertices.length * sizeof(float3), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(device_tempPointCloud.normals.array, device_pointCloud.normals.array,
        device_pointCloud.normals.length * sizeof(float3), cudaMemcpyDeviceToDevice));

    // 7. Move points into respective bins
    unsigned int* device_nextIndexTableEntries;
    checkCudaErrors(cudaMalloc(&device_nextIndexTableEntries, totalBinCount * sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(device_nextIndexTableEntries, device_indexTable,
                               totalBinCount * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    rearrangePointCloud<<<(device_pointCloud.vertices.length / 256) + 1, 256>>>(
            device_tempPointCloud, device_pointCloud,
            boundingBox,
            device_nextIndexTableEntries,
            binCounts, binSize);
    checkCudaErrors(cudaFree(device_nextIndexTableEntries));

    // 8. Delete temporary vertex buffer
    device_tempPointCloud.free();

    // 8. Count nearby points using new array and its index structure
    SpinImage::array<unsigned int> device_pointCountArray = {sampleCount, nullptr};
    checkCudaErrors(cudaMalloc(&device_pointCountArray.content, sampleCount * sizeof(unsigned int)));
    computePointCounts<<<sampleCount, 32>>>(
            device_pointCountArray, device_pointCloud, boundingBox, device_indexTable, binCounts, binSize, pointDensityRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaFree(device_indexTable);

    /*unsigned int* host_pointCountArray = new unsigned int[sampleCount];
    cudaMemcpy(host_pointCountArray, device_pointCountArray.content, sampleCount * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < sampleCount; i++) {
        std::cout << host_pointCountArray[i];
        if(i % 10 == 0) {
            std::cout << std::endl;
        } else {
            std::cout << ", ";
        }
    }*/

    return device_pointCountArray;
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

    SpinImage::array<unsigned int> device_pointCountArray =
            computePointDensities(pointDensityRadius, device_pointCloud, sampleCount);

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


