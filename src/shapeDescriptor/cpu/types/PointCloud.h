#pragma once

#include "float3.h"
#include "uchar4.h"

namespace ShapeDescriptor {
    namespace cpu {
        struct PointCloud {
            ShapeDescriptor::cpu::float3 *vertices = nullptr;
            ShapeDescriptor::cpu::float3 *normals = nullptr;
            ShapeDescriptor::cpu::uchar4 *vertexColours = nullptr;
            size_t pointCount = 0;

            PointCloud() = default;

            PointCloud(size_t pointCount) {
                vertices = new ShapeDescriptor::cpu::float3[pointCount];
                normals = new ShapeDescriptor::cpu::float3[pointCount];
                vertexColours = new ShapeDescriptor::cpu::uchar4[pointCount];
                pointCount = pointCount;
            }
        };
    }
}