#include <shapeDescriptor/utilities/free/mesh.h>

void ShapeDescriptor::free::mesh(ShapeDescriptor::cpu::Mesh meshToFree) {
    if(meshToFree.vertices != nullptr) {
        delete[] meshToFree.vertices;
        meshToFree.vertices = nullptr;
    }

    if(meshToFree.normals != nullptr) {
        delete[] meshToFree.normals;
        meshToFree.normals = nullptr;
    }
}

void ShapeDescriptor::free::mesh(ShapeDescriptor::gpu::Mesh meshToFree) {
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
}