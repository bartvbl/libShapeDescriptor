#pragma once

#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <array>
#include <spinImage/gpu/types/methods/FPFHDescriptor.h>

namespace SpinImage {
    namespace debug {
        struct FPFHExecutionTimes {
            double totalExecutionTimeSeconds;
            double originReformatExecutionTimeSeconds;
            double originSPFHGenerationExecutionTimeSeconds;
            double pointCloudSPFHGenerationExecutionTimeSeconds;
            double fpfhGenerationExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::gpu::array<SpinImage::gpu::FPFHDescriptor> generateFPFHHistograms(
                SpinImage::gpu::PointCloud device_pointCloud,
                SpinImage::gpu::array<DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                SpinImage::debug::FPFHExecutionTimes* executionTimes = nullptr);
    }
}