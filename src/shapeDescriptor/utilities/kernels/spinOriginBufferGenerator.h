#pragma once

#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <vector>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(gpu::Mesh &mesh);
        ShapeDescriptor::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(std::vector<cpu::float3> &vertices, std::vector<cpu::float3> &normals);
    }
}