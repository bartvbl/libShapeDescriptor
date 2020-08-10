#pragma once

#include "types/Mesh.h"
#include "types/DeviceOrientedPoint.h"
#include "types/array.h"
#include "shapeDescriptor/common/types/methods/RICIDescriptor.h"

namespace SpinImage {
    namespace debug {
        struct RICIExecutionTimes {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> generateRadialIntersectionCountImages(
                SpinImage::gpu::Mesh device_mesh,
                SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                SpinImage::debug::RICIExecutionTimes* executionTimes = nullptr);
    }
}