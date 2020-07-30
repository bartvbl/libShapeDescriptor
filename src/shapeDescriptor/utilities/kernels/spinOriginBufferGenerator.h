#pragma once

#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <vector>

namespace SpinImage {
    namespace utilities {
        SpinImage::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(gpu::Mesh &mesh);
        SpinImage::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(std::vector<cpu::float3> &vertices, std::vector<cpu::float3> &normals);
    }
}