#pragma once

#include <shapeDescriptor/gpu/types/VertexList.cuh>

namespace ShapeDescriptor {
    namespace cpu {
        struct PointCloud {
            float3 *vertices = nullptr;
            float3 *normals = nullptr;
            uchar4 *vertexColours = nullptr;
            size_t pointCount = 0;

            PointCloud() = default;

            PointCloud(size_t pointCount) {
                vertices = new float3[pointCount];
                normals = new float3[pointCount];
                vertexColours = new uchar4[pointCount];
                pointCount = pointCount;
            }
        };
    }
}