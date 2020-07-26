#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

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
        array<radialIntersectionCountImagePixelType> generateRadialIntersectionCountImages(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float spinImageWidth,
                SpinImage::debug::RICIExecutionTimes* executionTimes = nullptr);
    }
}