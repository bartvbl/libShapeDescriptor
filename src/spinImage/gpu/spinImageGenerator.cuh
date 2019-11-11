#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/Mesh.h>
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
        // A seed of 0 will cause the implementation to pick one
        array<spinImagePixelType> generateSpinImages(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float spinImageWidth,
                size_t sampleCount,
                float supportAngleDegrees,
                size_t randomSamplingSeed = 0,
                SpinImage::debug::SIRunInfo* runInfo = nullptr);
    }
}