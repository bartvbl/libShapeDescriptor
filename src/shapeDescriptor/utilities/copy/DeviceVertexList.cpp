#include <cuda_runtime_api.h>
#include <iostream>
#include "DeviceVertexList.h"

ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> ShapeDescriptor::copy::deviceVertexListToHost(ShapeDescriptor::gpu::DeviceVertexList vertexList) {
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> outList;
    outList.length = vertexList.length;
    outList.content = new ShapeDescriptor::cpu::float3[vertexList.length];

    size_t bufferSize = 3 * sizeof(float) * vertexList.length;

    float* tempVertexBuffer = new float[3 * vertexList.length];

    checkCudaErrors(cudaMemcpy(tempVertexBuffer, vertexList.array, bufferSize, cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < vertexList.length; i++) {
        outList.content[i] = {
                tempVertexBuffer[i + 0 * vertexList.length],
                tempVertexBuffer[i + 1 * vertexList.length],
                tempVertexBuffer[i + 2 * vertexList.length]};
    }

    delete[] tempVertexBuffer;

    return outList;
}