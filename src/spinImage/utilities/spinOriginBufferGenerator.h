#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/cpu/types/float3.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/cpu/types/Mesh.h>
#include <vector>

namespace SpinImage {
    namespace utilities {
        array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(gpu::Mesh &mesh);
        array<gpu::DeviceOrientedPoint> generateUniqueSpinOriginBuffer(std::vector<cpu::float3> &vertices, std::vector<cpu::float3> &normals);
    }
}