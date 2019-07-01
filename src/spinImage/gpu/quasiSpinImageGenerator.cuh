#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/DeviceMesh.h>


namespace SpinImage {
    namespace debug {
        struct QSIRunInfo {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<quasiSpinImagePixelType> generateQuasiSpinImages(
                DeviceMesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float spinImageWidth,
                SpinImage::debug::QSIRunInfo* runinfo = nullptr);
    }
}