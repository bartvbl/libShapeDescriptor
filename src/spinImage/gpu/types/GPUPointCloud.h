#pragma once

#include <spinImage/gpu/types/DeviceVertexList.cuh>

namespace SpinImage {
    struct GPUPointCloud {
        DeviceVertexList vertices;
        DeviceVertexList normals;

        GPUPointCloud(size_t pointCount) : vertices(pointCount), normals(pointCount) {}

        void free() {
            vertices.free();
            normals.free();
        }
    };
}