#pragma once

#include <spinImage/cpu/types/float3.h>
#include <spinImage/gpu/types/DeviceVertexList.cuh>

namespace SpinImage {
    namespace copy {
        SpinImage::array<SpinImage::cpu::float3> deviceVertexListToHost(SpinImage::gpu::DeviceVertexList vertexList);
    }
}