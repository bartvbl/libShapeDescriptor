#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>

namespace SpinImage {
    namespace debug {
        struct SCRunInfo {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double meshSamplingTimeSeconds;
            double generationTimeSeconds;
        };
    }

    namespace gpu {
        // A seed of 0 will cause the implementation to pick one
        array<shapeContextBinType > generate3DSCDescriptors(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_spinImageOrigins,
                float supportRadius,
                size_t sampleCount,
                size_t randomSamplingSeed = 0,
                SpinImage::debug::SCRunInfo* runInfo = nullptr);
    }
}