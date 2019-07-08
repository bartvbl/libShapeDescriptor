#include <iostream>
#include <cuda_runtime.h>
#include "spinOriginBufferGenerator.h"
#include <nvidia/helper_cuda.h>
#include <assert.h>
#include <spinImage/utilities/copy/hostMeshToDevice.h>

__global__ void removeDuplicates(DeviceMesh inputMesh, DeviceOrientedPoint* compactedOrigins, size_t* totalVertexCount) {
    // Only a single warp to avoid complications related to divergence within a block
    // (syncthreads may hang indefinitely if some threads diverged)
    const int threadCount = 32;

    // Kernel is made for a single block of threads for easy implementation
    assert(gridDim.x == 1 && gridDim.y == 1 && gridDim.z == 1);
    assert(blockDim.x == threadCount && blockDim.y == 1 && blockDim.z == 1);

    int threadIndex = threadIdx.x;

    __shared__ size_t arrayPointer;

    arrayPointer = 0;

    for(size_t vertexIndex = threadIndex; vertexIndex < inputMesh.vertexCount; vertexIndex += threadCount) {
        float3 vertex = make_float3(
                inputMesh.vertices_x[vertexIndex],
                inputMesh.vertices_y[vertexIndex],
                inputMesh.vertices_z[vertexIndex]);
        float3 normal = make_float3(
                inputMesh.normals_x[vertexIndex],
                inputMesh.normals_y[vertexIndex],
                inputMesh.normals_z[vertexIndex]);

        bool shouldBeDiscarded = false;

        for(size_t otherIndex = 0; otherIndex < vertexIndex; otherIndex++) {
            float3 otherVertex = make_float3(
                    inputMesh.vertices_x[otherIndex],
                    inputMesh.vertices_y[otherIndex],
                    inputMesh.vertices_z[otherIndex]);
            float3 otherNormal = make_float3(
                    inputMesh.normals_x[otherIndex],
                    inputMesh.normals_y[otherIndex],
                    inputMesh.normals_z[otherIndex]);

            // We're looking for exact matches here. Given that vertex duplications should
            // yield equivalent vertex coordinates, testing floating point numbers for
            // exact equivalence is warranted.
            if( vertex.x == otherVertex.x &&
                vertex.y == otherVertex.y &&
                vertex.z == otherVertex.z &&
                normal.x == otherNormal.x &&
                normal.y == otherNormal.y &&
                normal.z == otherNormal.z) {

                shouldBeDiscarded = true;
                break;
            }
        }

        __syncthreads();

        unsigned int uniqueVerticesInWarp = __ballot_sync(0xFFFFFFFF, !shouldBeDiscarded);
        unsigned int uniqueVertexCount = __popc(uniqueVerticesInWarp);

        unsigned int indicesBeforeMe = __popc(uniqueVerticesInWarp << (32 - threadIndex));
        size_t outVertexIndex = arrayPointer + indicesBeforeMe;

        if(!shouldBeDiscarded) {
            DeviceOrientedPoint spinOrigin;

            spinOrigin.vertex = vertex;
            spinOrigin.normal = normal;

            compactedOrigins[outVertexIndex] = spinOrigin;
        }

        if(threadIndex == 0) {
            arrayPointer += uniqueVertexCount;
        }
    }

    __syncthreads();

    // Returning the new size
    if(threadIndex == 0) {
        *totalVertexCount = arrayPointer;
    }
}

array<DeviceOrientedPoint> removeDuplicates(DeviceMesh mesh) {
    size_t* device_totalVertexCount;
    checkCudaErrors(cudaMalloc(&device_totalVertexCount, sizeof(size_t)));

    array<DeviceOrientedPoint> device_spinOrigins;
    checkCudaErrors(cudaMalloc(&device_spinOrigins.content, mesh.vertexCount * sizeof(DeviceOrientedPoint)));

    removeDuplicates<<<1, 32>>>(mesh, device_spinOrigins.content, device_totalVertexCount);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t totalVertexCount = 0;
    checkCudaErrors(cudaMemcpy(&totalVertexCount, device_totalVertexCount, sizeof(size_t), cudaMemcpyDeviceToHost));

    device_spinOrigins.length = totalVertexCount;

    checkCudaErrors(cudaFree(device_totalVertexCount));

    return device_spinOrigins;
}

array<DeviceOrientedPoint> SpinImage::utilities::generateUniqueSpinOriginBuffer(DeviceMesh &mesh) {
    return removeDuplicates(mesh);
}

array<DeviceOrientedPoint> SpinImage::utilities::generateUniqueSpinOriginBuffer(std::vector<float3_cpu> &vertices, std::vector<float3_cpu> &normals) {
    assert(vertices.size() == normals.size());

    float3_cpu* vertexData = vertices.data();
    float3_cpu* normalData = normals.data();

    unsigned int* indexBuffer = new unsigned int[vertices.size()];

    for(unsigned int i = 0; i < vertices.size(); i++) {
        indexBuffer[i] = i;
    }

    HostMesh tempHostMesh;

    tempHostMesh.vertices = vertexData;
    tempHostMesh.normals = normalData;
    tempHostMesh.indices = indexBuffer;
    tempHostMesh.vertexCount = vertices.size();
    tempHostMesh.indexCount = vertices.size();

    DeviceMesh tempDeviceMesh = SpinImage::copy::hostMeshToDevice(tempHostMesh);

    array<DeviceOrientedPoint> spinOrigins = removeDuplicates(tempDeviceMesh);

    SpinImage::gpu::freeDeviceMesh(tempDeviceMesh);
    delete[] indexBuffer;

    return spinOrigins;
}