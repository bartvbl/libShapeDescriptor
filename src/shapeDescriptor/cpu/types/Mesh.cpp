#include "Mesh.h"
#include <shapeDescriptor/utilities/copy/mesh.h>

ShapeDescriptor::cpu::Mesh ShapeDescriptor::cpu::Mesh::clone() const {
    ShapeDescriptor::cpu::Mesh copiedMesh(vertexCount);

    std::copy(vertices, vertices + vertexCount, copiedMesh.vertices);
    std::copy(normals, normals + vertexCount, copiedMesh.normals);
    std::copy(vertexColours, vertexColours + vertexCount, copiedMesh.vertexColours);

    return copiedMesh;
}

ShapeDescriptor::gpu::Mesh ShapeDescriptor::cpu::Mesh::copyToGPU() {
    return ShapeDescriptor::copy::hostMeshToDevice(*this);
}