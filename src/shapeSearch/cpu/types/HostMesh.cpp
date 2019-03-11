#include "HostMesh.h"

void freeHostMesh(HostMesh &mesh) {
    delete[] mesh.vertices;
    delete[] mesh.normals;
    delete[] mesh.indices;
}