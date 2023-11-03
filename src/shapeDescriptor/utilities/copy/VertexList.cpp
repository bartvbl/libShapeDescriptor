#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime_api.h>
#endif

ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> ShapeDescriptor::copyToCPU(ShapeDescriptor::gpu::VertexList vertexList) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> outList(vertexList.length);

    size_t bufferSize = 3 * sizeof(float) * vertexList.length;

    float* tempVertexBuffer = new float[3 * vertexList.length];

    checkCudaErrors(cudaMemcpy(tempVertexBuffer, vertexList.array, bufferSize, cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < vertexList.length; i++) {
        outList[i] = {
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

ShapeDescriptor::gpu::VertexList ShapeDescriptor::copyToGPU(ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::float3> hostArray) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    ShapeDescriptor::gpu::VertexList device_list(hostArray.length);
    ShapeDescriptor::cpu::array<float> rearrangementArray(3 * hostArray.length);

    for(size_t i = 0; i < hostArray.length; i++) {
        rearrangementArray.content[i + 0 * hostArray.length] = hostArray.content[i].x;
        rearrangementArray.content[i + 1 * hostArray.length] = hostArray.content[i].y;
        rearrangementArray.content[i + 2 * hostArray.length] = hostArray.content[i].z;
    }

    checkCudaErrors(cudaMemcpy(device_list.array, rearrangementArray.content, 3 * hostArray.length * sizeof(float), cudaMemcpyHostToDevice));

    ShapeDescriptor::free(rearrangementArray);

    return device_list;
#else
    throw std::runtime_error(cudaMissingErrorMessage);
#endif
}


