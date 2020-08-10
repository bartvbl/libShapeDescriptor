#include "Mesh.h"

void ShapeDescriptor::cpu::freeMesh(ShapeDescriptor::cpu::Mesh &mesh) {
    delete[] mesh.vertices;
    delete[] mesh.normals;
    delete[] mesh.indices;
}