#pragma once

#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/cpu/types/float3.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/cpu/types/Mesh.h>
#include <spinImage/gpu/types/array.h>
#include <vector>

namespace SpinImage {
    namespace utilities {
        SpinImage::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(gpu::Mesh &mesh);
        SpinImage::gpu::array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(std::vector<cpu::float3> &vertices, std::vector<cpu::float3> &normals);
    }
}