#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/common/types/methods/FPFHDescriptor.h>

namespace ShapeDescriptor {
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
        ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> generateFPFHHistograms(
                ShapeDescriptor::gpu::PointCloud device_pointCloud,
                ShapeDescriptor::gpu::array<DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::debug::FPFHExecutionTimes* executionTimes = nullptr);
    }
}