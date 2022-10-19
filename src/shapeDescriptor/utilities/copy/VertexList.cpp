#include "VertexList.h"
#include <iostream>
#include <shapeDescriptor/utilities/free/array.h>
#include <shapeDescriptor/libraryBuildSettings.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#endif

ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> ShapeDescriptor::copy::deviceVertexListToHost(ShapeDescriptor::gpu::VertexList vertexList) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
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
#else
    throw std::runtime_error(cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::gpu::VertexList ShapeDescriptor::copy::hostVertexListToDevice(ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> hostArray) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::VertexList device_list(hostArray.length);
    ShapeDescriptor::cpu::array<float> rearrangementArray(3 * hostArray.length);

    for(size_t i = 0; i < hostArray.length; i++) {
        rearrangementArray.content[i + 0 * hostArray.length] = hostArray.content[i].x;
        rearrangementArray.content[i + 1 * hostArray.length] = hostArray.content[i].y;
        rearrangementArray.content[i + 2 * hostArray.length] = hostArray.content[i].z;
    }

    checkCudaErrors(cudaMemcpy(device_list.array, rearrangementArray.content, 3 * hostArray.length * sizeof(float), cudaMemcpyHostToDevice));

    ShapeDescriptor::free::array(rearrangementArray);

    return device_list;
#else
    throw std::runtime_error(cudaMissingErrorMessage);
#endif
}