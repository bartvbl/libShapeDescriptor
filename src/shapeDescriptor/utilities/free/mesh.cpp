#include <shapeDescriptor/shapeDescriptor.h>

void ShapeDescriptor::free(ShapeDescriptor::cpu::Mesh &meshToFree) {
    if(meshToFree.vertices != nullptr) {
        delete[] meshToFree.vertices;
        meshToFree.vertices = nullptr;
    }

    if(meshToFree.normals != nullptr) {
        delete[] meshToFree.normals;
        meshToFree.normals = nullptr;
    }

    if(meshToFree.vertexColours != nullptr) {
        delete[] meshToFree.vertexColours;
        meshToFree.vertexColours = nullptr;
    }
}

void ShapeDescriptor::free(ShapeDescriptor::gpu::Mesh &meshToFree) {
CUDA_REGION(
    if(meshToFree.vertices_x != nullptr) {
        cudaFree(meshToFree.vertices_x);
        cudaFree(meshToFree.vertices_y);
        cudaFree(meshToFree.vertices_z);

        meshToFree.vertices_x = nullptr;
        meshToFree.vertices_y = nullptr;
        meshToFree.vertices_z = nullptr;
    }

    if(meshToFree.normals_x != nullptr) {
        cudaFree(meshToFree.normals_x);
        cudaFree(meshToFree.normals_y);
        cudaFree(meshToFree.normals_z);

        meshToFree.normals_x = nullptr;
        meshToFree.normals_y = nullptr;
        meshToFree.normals_z = nullptr;
    }
)
}