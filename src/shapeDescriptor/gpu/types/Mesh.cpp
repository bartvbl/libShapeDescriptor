#include <stdexcept>
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif


ShapeDescriptor::gpu::Mesh ShapeDescriptor::gpu::duplicateMesh(Mesh mesh) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t bufferSize = mesh.vertexCount * sizeof(float);

    Mesh outMesh;

    outMesh.vertexCount = mesh.vertexCount;

    checkCudaErrors(cudaMalloc(&outMesh.normals_x, bufferSize));
    checkCudaErrors(cudaMalloc(&outMesh.normals_y, bufferSize));
    checkCudaErrors(cudaMalloc(&outMesh.normals_z, bufferSize));

    checkCudaErrors(cudaMalloc(&outMesh.vertices_x, bufferSize));
    checkCudaErrors(cudaMalloc(&outMesh.vertices_y, bufferSize));
    checkCudaErrors(cudaMalloc(&outMesh.vertices_z, bufferSize));

    checkCudaErrors(cudaMemcpy(outMesh.normals_x, mesh.normals_x, bufferSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(outMesh.normals_y, mesh.normals_y, bufferSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(outMesh.normals_z, mesh.normals_z, bufferSize, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemcpy(outMesh.vertices_x, mesh.vertices_x, bufferSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(outMesh.vertices_y, mesh.vertices_y, bufferSize, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(outMesh.vertices_z, mesh.vertices_z, bufferSize, cudaMemcpyDeviceToDevice));

    return outMesh;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}


