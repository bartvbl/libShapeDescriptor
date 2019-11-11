#include "Mesh.h"

void SpinImage::cpu::freeMesh(SpinImage::cpu::Mesh &mesh) {
    delete[] mesh.vertices;
    delete[] mesh.normals;
    delete[] mesh.indices;
}