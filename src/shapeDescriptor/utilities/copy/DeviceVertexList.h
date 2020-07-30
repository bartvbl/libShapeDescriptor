#pragma once

#include <spinImage/cpu/types/float3.h>
#include <spinImage/gpu/types/DeviceVertexList.cuh>
#include <spinImage/cpu/types/array.h>

namespace SpinImage {
    namespace copy {
        SpinImage::cpu::array<SpinImage::cpu::float3> deviceVertexListToHost(SpinImage::gpu::DeviceVertexList vertexList);
    }
}