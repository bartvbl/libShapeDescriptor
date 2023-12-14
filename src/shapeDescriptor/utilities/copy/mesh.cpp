
#include <shapeDescriptor/shapeDescriptor.h>

#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include <cuda_runtime.h>
#include <helper_cuda.h>
#endif

ShapeDescriptor::gpu::Mesh ShapeDescriptor::copyToGPU(cpu::Mesh hostMesh)
{
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t vertexCount = hostMesh.vertexCount;
    size_t normalCount = hostMesh.vertexCount;

    float* device_vertices_x = nullptr;
    float* device_vertices_y = nullptr;
    float* device_vertices_z = nullptr;

    float* device_normals_x = nullptr;
    float* device_normals_y = nullptr;
    float* device_normals_z = nullptr;

    size_t verticesSize = sizeof(float) * vertexCount;
    size_t normalsSize = sizeof(float) * normalCount;

    checkCudaErrors(cudaMalloc(&device_vertices_x, verticesSize));
    checkCudaErrors(cudaMalloc(&device_vertices_y, verticesSize));
    checkCudaErrors(cudaMalloc(&device_vertices_z, verticesSize));

    checkCudaErrors(cudaMalloc(&device_normals_x, normalsSize));
    checkCudaErrors(cudaMalloc(&device_normals_y, normalsSize));
    checkCudaErrors(cudaMalloc(&device_normals_z, normalsSize));

    // Because the GPU requires a different vertex format, we need to decompose
    // the vertex and normal arrays into their separate components.
    float* vertices_x = new float[vertexCount];
    float* vertices_y = new float[vertexCount];
    float* vertices_z = new float[vertexCount];

    float* normals_x = new float[normalCount];
    float* normals_y = new float[normalCount];
    float* normals_z = new float[normalCount];

    for(unsigned int i = 0; i < vertexCount; i++) {
        vertices_x[i] = hostMesh.vertices[i].x;
        vertices_y[i] = hostMesh.vertices[i].y;
        vertices_z[i] = hostMesh.vertices[i].z;
    }

    for(int i = 0; i < normalCount; i++) {
        normals_x[i] = hostMesh.normals[i].x;
        normals_y[i] = hostMesh.normals[i].y;
        normals_z[i] = hostMesh.normals[i].z;
    }

    // Copy input vectors from host memory to GPU buffers.
    checkCudaErrors(cudaMemcpy(device_vertices_x, vertices_x, verticesSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vertices_y, vertices_y, verticesSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_vertices_z, vertices_z, verticesSize, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(device_normals_x, normals_x, normalsSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_normals_y, normals_y, normalsSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(device_normals_z, normals_z, normalsSize, cudaMemcpyHostToDevice));

    // Delete temporary buffers
    delete[] vertices_x;
    delete[] vertices_y;
    delete[] vertices_z;

    delete[] normals_x;
    delete[] normals_y;
    delete[] normals_z;

    // Construct the mesh struct for the GPU side
    gpu::Mesh device_mesh;

    device_mesh.vertices_x = device_vertices_x;
    device_mesh.vertices_y = device_vertices_y;
    device_mesh.vertices_z = device_vertices_z;

    device_mesh.normals_x = device_normals_x;
    device_mesh.normals_y = device_normals_y;
    device_mesh.normals_z = device_normals_z;

    device_mesh.vertexCount = hostMesh.vertexCount;

    return device_mesh;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::cpu::Mesh ShapeDescriptor::copyToCPU(gpu::Mesh deviceMesh)
{
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
    size_t vertexCount = deviceMesh.vertexCount;

    cpu::Mesh hostMesh(vertexCount);

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

    return hostMesh;
#else
    throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
}

ShapeDescriptor::cpu::Mesh ShapeDescriptor::gpu::Mesh::copyToCPU() {
    return ShapeDescriptor::copyToCPU(*this);
}