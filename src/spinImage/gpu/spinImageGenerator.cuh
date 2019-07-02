#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/DeviceMesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

namespace SpinImage {
    namespace debug {
        struct SIRunInfo {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double meshSamplingTimeSeconds;
            double generationTimeSeconds;
        };
    }

    namespace gpu {
        array<spinImagePixelType> generateSpinImages(
                DeviceMesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float spinImageWidth,
                size_t sampleCount,
                float supportAngleDegrees,
                SpinImage::debug::SIRunInfo* runInfo = nullptr);
    }
}