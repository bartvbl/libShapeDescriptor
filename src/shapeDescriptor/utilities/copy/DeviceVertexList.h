#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <shapeDescriptor/gpu/types/DeviceVertexList.cuh>
#include <shapeDescriptor/cpu/types/array.h>

namespace SpinImage {
    namespace copy {
        SpinImage::cpu::array<SpinImage::cpu::float3> deviceVertexListToHost(SpinImage::gpu::DeviceVertexList vertexList);
    }
}