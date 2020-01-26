#include <nvidia/helper_cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <limits>
#include "pointCloudUtils.h"

// Not a super efficient implementation, but good enough for what we need it to do.

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

__global__ void computePointCloudBoundingBox(
        SpinImage::gpu::PointCloud pointCloud,
        SpinImage::gpu::BoundingBox* boundingBox) {

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

SpinImage::gpu::BoundingBox SpinImage::utilities::computeBoundingBox(SpinImage::gpu::PointCloud device_pointCloud) {
    SpinImage::gpu::BoundingBox host_boundingBox = {
            {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
            {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()}};
    SpinImage::gpu::BoundingBox* device_boundingBox;
    checkCudaErrors(cudaMalloc(&device_boundingBox, sizeof(SpinImage::gpu::BoundingBox)));
    checkCudaErrors(cudaMemcpy(device_boundingBox, &host_boundingBox, sizeof(SpinImage::gpu::BoundingBox), cudaMemcpyHostToDevice));

    // Single block, because CUDA is not being nice to me.
    computePointCloudBoundingBox<<<1, 1024>>>(device_pointCloud, device_boundingBox);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&host_boundingBox, device_boundingBox, sizeof(SpinImage::gpu::BoundingBox), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_boundingBox));
    return host_boundingBox;
}