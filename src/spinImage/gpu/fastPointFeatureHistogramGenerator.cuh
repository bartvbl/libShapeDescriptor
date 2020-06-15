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
        // Clone of the pcl::FPFHSignature33 struct
        // by duplicating it, we ensure that a library using this one does not need to have PCL headers available.
        struct FPFHHistogram33 {
            float contents[33];
        };

        struct FPFHHistograms {
            FPFHHistogram33* histograms;
            unsigned int count;
        };

        // A seed of 0 will cause the implementation to pick one
        FPFHHistograms generateFPFHHistograms(
                Mesh device_mesh,
                array<DeviceOrientedPoint> device_origins,
                float supportRadius = 0.2,
                unsigned int maxNeighbours = 50,
                size_t sampleCount = 1000000,
                size_t randomSamplingSeed = 0,
                SpinImage::debug::FPFHRunInfo* runInfo = nullptr);
    }
}