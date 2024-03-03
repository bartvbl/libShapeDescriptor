#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <helper_cuda.h>
#include <cuda_runtime.h>
#include <helper_math.h>
#endif

#include <cfloat>
#include <limits>
#include <iostream>

// -- Utility functions --

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val = __shfl_xor_sync(0xFFFFFFFF, val, mask) + val;
    return val;
}

__inline__ __device__ float warpAllReduceMin(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val = min(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    return val;
}

__inline__ __device__ float warpAllReduceMax(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val = max(__shfl_xor_sync(0xFFFFFFFF, val, mask), val);
    return val;
}

std::ostream& operator << (std::ostream &o, const float3& p)
{
    return o << "(" << p.x << ", " << p.y << ", " << p.z << ")";
}




// Not a super efficient implementation, but good enough for what we need it to do.

__global__ void computePointCloudBoundingBox(
        ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeDescriptor::gpu::BoundingBox* boundingBox) {

    assert(blockDim.x == 1024);
    __shared__ float3 minVertices[1024 / 32];
    __shared__ float3 maxVertices[1024 / 32];

    float3 minVertex = {FLT_MAX, FLT_MAX, FLT_MAX};
    float3 maxVertex = {-FLT_MAX, -FLT_MAX, -FLT_MAX};

    for(size_t vertexIndex = threadIdx.x; vertexIndex < pointCloud.vertices.length; vertexIndex += blockDim.x) {
        float3 vertex = pointCloud.vertices.at(vertexIndex);

        minVertex = {
            min(minVertex.x, vertex.x),
            min(minVertex.y, vertex.y),
            min(minVertex.z, vertex.z)
        };

        maxVertex = {
            max(maxVertex.x, vertex.x),
            max(maxVertex.y, vertex.y),
            max(maxVertex.z, vertex.z)
        };
    }

    float3 totalMinVertex = {
        warpAllReduceMin(minVertex.x),
        warpAllReduceMin(minVertex.y),
        warpAllReduceMin(minVertex.z)
    };

    float3 totalMaxVertex = {
        warpAllReduceMax(maxVertex.x),
        warpAllReduceMax(maxVertex.y),
        warpAllReduceMax(maxVertex.z)
    };

    if(threadIdx.x % 32 == 0) {
        minVertices[threadIdx.x / 32] = totalMinVertex;
        maxVertices[threadIdx.x / 32] = totalMaxVertex;
    }

    __syncthreads();

    if(threadIdx.x < 32) {
        for(int i = threadIdx.x; i < blockDim.x / 32; i += 32) {
            totalMinVertex = {
                min(totalMinVertex.x, warpAllReduceMin(minVertices[i].x)),
                min(totalMinVertex.y, warpAllReduceMin(minVertices[i].y)),
                min(totalMinVertex.z, warpAllReduceMin(minVertices[i].z))
            };
            totalMaxVertex = {
                max(totalMaxVertex.x, warpAllReduceMin(maxVertices[i].x)),
                max(totalMaxVertex.y, warpAllReduceMin(maxVertices[i].y)),
                max(totalMaxVertex.z, warpAllReduceMin(maxVertices[i].z))
            };
        }

        if(threadIdx.x == 0) {
            boundingBox->min = totalMinVertex;
            boundingBox->max = totalMaxVertex;
        }
    }

}
#endif

ShapeDescriptor::gpu::BoundingBox ShapeDescriptor::computeBoundingBox(ShapeDescriptor::gpu::PointCloud device_pointCloud) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::BoundingBox host_boundingBox = {
            {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
            {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}};
    ShapeDescriptor::gpu::BoundingBox* device_boundingBox;
    checkCudaErrors(cudaMalloc(&device_boundingBox, sizeof(ShapeDescriptor::gpu::BoundingBox)));
    checkCudaErrors(cudaMemcpy(device_boundingBox, &host_boundingBox, sizeof(ShapeDescriptor::gpu::BoundingBox), cudaMemcpyHostToDevice));

    // Single block, because CUDA is not being nice to me.
    computePointCloudBoundingBox<<<1, 1024>>>(device_pointCloud, device_boundingBox);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&host_boundingBox, device_boundingBox, sizeof(ShapeDescriptor::gpu::BoundingBox), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_boundingBox));
    return host_boundingBox;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}






















#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__device__ __inline__ unsigned int computeJumpTableIndex(int3 binIndex, int3 binCounts) {
    return binIndex.z * binCounts.x * binCounts.y + binIndex.y * binCounts.x + binIndex.x;
}

__global__ void countBinContents(
        ShapeDescriptor::gpu::PointCloud pointCloud,
        unsigned int* indexTable,
        ShapeDescriptor::gpu::BoundingBox boundingBox,
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

    unsigned int indexTableIndex = computeJumpTableIndex(binIndex, binCounts);

    assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

    atomicAdd(&indexTable[indexTableIndex], 1);
}

__global__ void countCumulativeBinIndices(unsigned int* indexTable, int3 binCounts, unsigned int pointCloudSize) {
    unsigned int cumulativeIndex = 0;
    for(int z = 0; z < binCounts.z; z++) {
        for(int y = 0; y < binCounts.y; y++) {
            for(int x = 0; x < binCounts.x; x++) {
                unsigned int binIndex = computeJumpTableIndex({x, y, z}, binCounts);
                unsigned int binLength = indexTable[binIndex];
                indexTable[binIndex] = cumulativeIndex;
                cumulativeIndex += binLength;
            }
        }
    }
    assert(cumulativeIndex == pointCloudSize);
}

__global__ void rearrangePointCloud(
        ShapeDescriptor::gpu::PointCloud sourcePointCloud,
        ShapeDescriptor::gpu::PointCloud destinationPointCloud,
        ShapeDescriptor::gpu::BoundingBox boundingBox,
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

    unsigned int indexTableIndex = computeJumpTableIndex(binIndex, binCounts);

    assert(indexTableIndex < binCounts.x * binCounts.y * binCounts.z);

    unsigned int targetIndex = atomicAdd(&nextIndexEntryTable[indexTableIndex], 1);

    destinationPointCloud.vertices.set(targetIndex, vertex);
}

__global__ void computePointCounts(
        ShapeDescriptor::gpu::array<unsigned int> pointDensityArray,
        ShapeDescriptor::gpu::PointCloud pointCloud,
        ShapeDescriptor::gpu::BoundingBox boundingBox,
        unsigned int* indexTable,
        int3 binCounts,
        float binSize,
        float countRadius) {

    assert(blockDim.x == 32);

#define pointIndex blockIdx.x
    unsigned int threadPointCount = 0;
    float3 referencePoint = pointCloud.vertices.at(pointIndex);

    float3 referencePointBoundsMin = referencePoint - float3{countRadius, countRadius, countRadius} - boundingBox.min;
    float3 referencePointBoundsMax = referencePoint + float3{countRadius, countRadius, countRadius} - boundingBox.min;

    // Ensure coordinates range between 0 and length-1
    int3 minBinIndices = {
            min(max(int(referencePointBoundsMin.x / binSize) - 1, 0), binCounts.x-1),
            min(max(int(referencePointBoundsMin.y / binSize) - 1, 0), binCounts.y-1),
            min(max(int(referencePointBoundsMin.z / binSize) - 1, 0), binCounts.z-1)
    };

    int3 maxBinIndices = {
            min(max(int(referencePointBoundsMax.x / binSize) + 1, 0), binCounts.x-1),
            min(max(int(referencePointBoundsMax.y / binSize) + 1, 0), binCounts.y-1),
            min(max(int(referencePointBoundsMax.z / binSize) + 1, 0), binCounts.z-1)
    };

    assert(minBinIndices.x < binCounts.x);
    assert(minBinIndices.y < binCounts.y);
    assert(minBinIndices.z < binCounts.z);
    assert(maxBinIndices.x < binCounts.x);
    assert(maxBinIndices.y < binCounts.y);
    assert(maxBinIndices.z < binCounts.z);

    for(int binZ = minBinIndices.z; binZ <= maxBinIndices.z; binZ++) {
        for(int binY = minBinIndices.y; binY <= maxBinIndices.y; binY++) {
            unsigned int startTableIndex = computeJumpTableIndex({minBinIndices.x, binY, binZ}, binCounts);
            unsigned int endTableIndex = computeJumpTableIndex({maxBinIndices.x, binY, binZ}, binCounts) + 1;

            unsigned int startVertexIndex = indexTable[startTableIndex];
            unsigned int endVertexIndex = 0;

            if(endTableIndex < binCounts.x * binCounts.y * binCounts.z - 1) {
                endVertexIndex = indexTable[endTableIndex];
            } else {
                endVertexIndex = pointCloud.vertices.length;
            }

            assert(startVertexIndex <= endVertexIndex);
            assert(startVertexIndex <= pointCloud.vertices.length);
            assert(endVertexIndex <= pointCloud.vertices.length);

            for (unsigned int samplePointIndex = startVertexIndex + threadIdx.x; samplePointIndex < endVertexIndex; samplePointIndex += blockDim.x)
            {
                if(samplePointIndex == pointIndex) {
                    continue;
                }

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
#endif

ShapeDescriptor::gpu::array<unsigned int> ShapeDescriptor::computePointDensities(
        float pointDensityRadius, ShapeDescriptor::gpu::PointCloud device_pointCloud) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t sampleCount = device_pointCloud.vertices.length;

    // 1. Compute bounding box
    ShapeDescriptor::gpu::BoundingBox boundingBox = ShapeDescriptor::computeBoundingBox(device_pointCloud);

    // 2. Allocate index array for boxes of radius x radius x radius
    float3 boundingBoxSize = boundingBox.max - boundingBox.min;
    float binSize = std::cbrt(
            (boundingBoxSize.x != 0 ? boundingBoxSize.x : 1) *
               (boundingBoxSize.y != 0 ? boundingBoxSize.y : 1) *
               (boundingBoxSize.z != 0 ? boundingBoxSize.z : 1)) / 50.0f;

    int3 binCounts = {int(boundingBoxSize.x / binSize) + 1,
                      int(boundingBoxSize.y / binSize) + 1,
                      int(boundingBoxSize.z / binSize) + 1};
    int totalBinCount = binCounts.x * binCounts.y * binCounts.z;
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
    ShapeDescriptor::gpu::PointCloud device_tempPointCloud(device_pointCloud.vertices.length);

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
    ShapeDescriptor::gpu::array<unsigned int> device_pointCountArray = {sampleCount, nullptr};
    checkCudaErrors(cudaMalloc(&device_pointCountArray.content, sampleCount * sizeof(unsigned int)));
    computePointCounts<<<sampleCount, 32>>>(
            device_pointCountArray, device_pointCloud, boundingBox, device_indexTable, binCounts, binSize, pointDensityRadius);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    cudaFree(device_indexTable);

    return device_pointCountArray;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

