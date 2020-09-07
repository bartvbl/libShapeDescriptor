#pragma once

#include <shapeDescriptor/gpu/types/VertexList.cuh>

namespace ShapeDescriptor {
    namespace gpu {
        struct PointCloud {
            VertexList vertices;
            VertexList normals;
            size_t pointCount;

            PointCloud(size_t pointCount) : vertices(pointCount), normals(pointCount), pointCount(pointCount) {}

            void free() {
                vertices.free();
                normals.free();
            }
        };
    }


}