#pragma once

#include <spinImage/gpu/types/DeviceVertexList.cuh>

namespace SpinImage {
    namespace gpu {
        struct PointCloud {
            DeviceVertexList vertices;
            DeviceVertexList normals;

            PointCloud(size_t pointCount) : vertices(pointCount), normals(pointCount) {}

            void free() {
                vertices.free();
                normals.free();
            }
        };
    }


}