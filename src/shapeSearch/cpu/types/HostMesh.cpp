#include "HostMesh.h"

void SpinImage::cpu::freeHostMesh(HostMesh &mesh) {
    delete[] mesh.vertices;
    delete[] mesh.normals;
    delete[] mesh.indices;
}