#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/methods/RICIDescriptor.h>

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
        SpinImage::array<SpinImage::gpu::RICIDescriptor> generateRadialIntersectionCountImages(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                SpinImage::debug::RICIExecutionTimes* executionTimes = nullptr);
    }
}