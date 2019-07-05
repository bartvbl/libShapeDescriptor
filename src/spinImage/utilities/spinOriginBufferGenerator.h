#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/cpu/types/float3_cpu.h>
#include <vector>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/cpu/types/HostMesh.h>

namespace SpinImage {
    namespace utilities {
        array<DeviceOrientedPoint> generateUniqueSpinOriginBuffer(DeviceMesh &mesh);
        array<DeviceOrientedPoint> generateUniqueSpinOriginBuffer(std::vector<float3_cpu> &vertices, std::vector<float3_cpu> &normals);
    }
}