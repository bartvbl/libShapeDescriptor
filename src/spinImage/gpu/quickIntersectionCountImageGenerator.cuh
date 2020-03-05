#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/QUICCImages.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

namespace SpinImage {
    namespace debug {
        struct QUICCIRunInfo {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        QUICCIImages generateQUICCImages(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float supportRadius,
                SpinImage::debug::QUICCIRunInfo* runinfo = nullptr);
    }
}