#include <cuda_runtime.h>
#include "DeviceMesh.h"

void freeDeviceMesh(DeviceMesh mesh) {
    cudaFree(mesh.vertices_x);
    cudaFree(mesh.vertices_y);
    cudaFree(mesh.vertices_z);

    cudaFree(mesh.normals_x);
    cudaFree(mesh.normals_y);
    cudaFree(mesh.normals_z);
}