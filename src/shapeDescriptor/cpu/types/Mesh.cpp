#include "Mesh.h"

ShapeDescriptor::cpu::Mesh ShapeDescriptor::cpu::Mesh::clone() const {
    ShapeDescriptor::cpu::Mesh copiedMesh(vertexCount);

    std::copy(vertices, vertices + vertexCount, copiedMesh.vertices);
    std::copy(normals, normals + vertexCount, copiedMesh.normals);

    return copiedMesh;
}
