#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/methods/FPFHDescriptor.h>

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