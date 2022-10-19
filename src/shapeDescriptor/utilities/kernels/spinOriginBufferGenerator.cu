#include "spinOriginBufferGenerator.h"

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#endif

#include <iostream>
#include <cassert>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <shapeDescriptor/gpu/types/array.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void removeDuplicates(ShapeDescriptor::gpu::Mesh inputMesh, ShapeDescriptor::OrientedPoint* compactedOrigins, size_t* totalVertexCount) {
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

        unsigned int uniqueVerticesInWarp = __ballot_sync(__activemask(), !shouldBeDiscarded);
        unsigned int uniqueVertexCount = __popc(uniqueVerticesInWarp);

        unsigned int indicesBeforeMe = __popc(uniqueVerticesInWarp << (32 - threadIndex));
        size_t outVertexIndex = arrayPointer + indicesBeforeMe;

        if(!shouldBeDiscarded) {
            ShapeDescriptor::OrientedPoint spinOrigin;

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
#endif

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> removeDuplicates(ShapeDescriptor::gpu::Mesh mesh) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t* device_totalVertexCount;
    checkCudaErrors(cudaMalloc(&device_totalVertexCount, sizeof(size_t)));

    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_spinOrigins;
    checkCudaErrors(cudaMalloc(&device_spinOrigins.content, mesh.vertexCount * sizeof(ShapeDescriptor::OrientedPoint)));

    removeDuplicates<<<1, 32>>>(mesh, device_spinOrigins.content, device_totalVertexCount);
    checkCudaErrors(cudaDeviceSynchronize());

    size_t totalVertexCount = 0;
    checkCudaErrors(cudaMemcpy(&totalVertexCount, device_totalVertexCount, sizeof(size_t), cudaMemcpyDeviceToHost));

    device_spinOrigins.length = totalVertexCount;

    checkCudaErrors(cudaFree(device_totalVertexCount));

    return device_spinOrigins;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(gpu::Mesh &mesh) {
    return removeDuplicates(mesh);
}

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> ShapeDescriptor::utilities::generateUniqueSpinOriginBuffer(std::vector<cpu::float3> &vertices, std::vector<cpu::float3> &normals) {
    assert(vertices.size() == normals.size());

    cpu::float3* vertexData = vertices.data();
    cpu::float3* normalData = normals.data();

    unsigned int* indexBuffer = new unsigned int[vertices.size()];

    for(unsigned int i = 0; i < vertices.size(); i++) {
        indexBuffer[i] = i;
    }

    cpu::Mesh tempHostMesh;

    tempHostMesh.vertices = vertexData;
    tempHostMesh.normals = normalData;
    tempHostMesh.vertexCount = vertices.size();

    gpu::Mesh tempDeviceMesh = ShapeDescriptor::copy::hostMeshToDevice(tempHostMesh);

    ShapeDescriptor::gpu::array<OrientedPoint> spinOrigins = removeDuplicates(tempDeviceMesh);

    ShapeDescriptor::gpu::freeMesh(tempDeviceMesh);
    delete[] indexBuffer;

    return spinOrigins;
}


#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
__global__ void convertMeshIntoOriginsList(ShapeDescriptor::gpu::Mesh inputMesh, ShapeDescriptor::OrientedPoint* origins) {
    size_t vertexIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if(vertexIndex >= inputMesh.vertexCount) {
        return;
    }

    float3 vertex = make_float3(
            inputMesh.vertices_x[vertexIndex],
            inputMesh.vertices_y[vertexIndex],
            inputMesh.vertices_z[vertexIndex]);
    float3 normal = make_float3(
            inputMesh.normals_x[vertexIndex],
            inputMesh.normals_y[vertexIndex],
            inputMesh.normals_z[vertexIndex]);

    origins[vertexIndex] = {vertex, normal};
}
#endif

ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>
ShapeDescriptor::utilities::generateSpinOriginBuffer(ShapeDescriptor::gpu::Mesh &mesh) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_spinOrigins;
    checkCudaErrors(cudaMalloc(&device_spinOrigins.content, mesh.vertexCount * sizeof(ShapeDescriptor::OrientedPoint)));
    device_spinOrigins.length = mesh.vertexCount;

    convertMeshIntoOriginsList<<<(mesh.vertexCount / 32) + 1, 32>>>(mesh, device_spinOrigins.content);
    checkCudaErrors(cudaDeviceSynchronize());

    return device_spinOrigins;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}
