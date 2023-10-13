#include <shapeDescriptor/shapeDescriptor.h>

ShapeDescriptor::cpu::Mesh ShapeDescriptor::cpu::Mesh::clone() const {
    ShapeDescriptor::cpu::Mesh copiedMesh(vertexCount);

    std::copy(vertices, vertices + vertexCount, copiedMesh.vertices);
    if(normals != nullptr) {
        std::copy(normals, normals + vertexCount, copiedMesh.normals);
    } else {
        delete[] copiedMesh.normals;
        copiedMesh.normals = nullptr;
    }

    if(vertexColours != nullptr) {
        assert(copiedMesh.vertexColours != nullptr);
        std::copy(vertexColours, vertexColours + vertexCount, copiedMesh.vertexColours);
    } else {
        delete[] copiedMesh.vertexColours;
        copiedMesh.vertexColours = nullptr;
    }

    return copiedMesh;
}

ShapeDescriptor::gpu::Mesh ShapeDescriptor::cpu::Mesh::copyToGPU() {
    return ShapeDescriptor::copyToGPU(*this);
}