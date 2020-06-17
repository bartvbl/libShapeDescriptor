#pragma once

#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <array>

namespace SpinImage {
    namespace debug {
        struct FPFHRunInfo {
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        struct FPFHHistograms {
            float* histograms;
            unsigned int count;
            unsigned int binsPerHistogramFeature;
        };

        // A seed of 0 will cause the implementation to pick one
        FPFHHistograms generateFPFHHistograms(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_origins,
                float supportRadius = 0.2,
                unsigned int numDescriptorBinsPerAxis = 11,
                size_t sampleCount = 1000000,
                size_t randomSamplingSeed = 0,
                SpinImage::debug::FPFHRunInfo* runInfo = nullptr);
    }
}