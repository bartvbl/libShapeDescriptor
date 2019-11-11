#include "deviceMeshToHost.h"

#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/gpu/types/Mesh.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>

SpinImage::cpu::Mesh SpinImage::copy::deviceMeshToHost(gpu::Mesh deviceMesh)
{
    size_t vertexCount = deviceMesh.vertexCount;

    cpu::Mesh hostMesh(vertexCount, vertexCount);

    size_t verticesSize = sizeof(float) * vertexCount;

    float* tempVertexBuffer = new float[3 * vertexCount];

    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 0 * vertexCount, deviceMesh.vertices_x, verticesSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 1 * vertexCount, deviceMesh.vertices_y, verticesSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 2 * vertexCount, deviceMesh.vertices_z, verticesSize, cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < vertexCount; i++) {
        hostMesh.vertices[i] = {
                tempVertexBuffer[i + 0 * vertexCount],
                tempVertexBuffer[i + 1 * vertexCount],
                tempVertexBuffer[i + 2 * vertexCount]};
    }

    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 0 * vertexCount, deviceMesh.normals_x, verticesSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 1 * vertexCount, deviceMesh.normals_y, verticesSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(tempVertexBuffer + 2 * vertexCount, deviceMesh.normals_z, verticesSize, cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < vertexCount; i++) {
        hostMesh.normals[i] = {
                tempVertexBuffer[i + 0 * vertexCount],
                tempVertexBuffer[i + 1 * vertexCount],
                tempVertexBuffer[i + 2 * vertexCount]};
    }

    delete[] tempVertexBuffer;

    for(size_t i = 0; i < vertexCount; i++) {
        hostMesh.indices[i] = i;
    }

    return hostMesh;
}