#pragma once

#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/common/types/methods/3DSCDescriptor.h>

namespace ShapeDescriptor {
    namespace debug {
        struct SCExecutionTimes {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double generationTimeSeconds;
            double pointCountingTimeSeconds;
        };
    }

    namespace internal {
        float computeBinVolume(short verticalBinIndex, short layerIndex, float minSupportRadius, float maxSupportRadius);
    }

    namespace gpu {
        // A seed of 0 will cause the implementation to pick one
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::ShapeContextDescriptor> generate3DSCDescriptors(
                ShapeDescriptor::gpu::PointCloud device_pointCloud,
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                float pointDensityRadius,
                float minSupportRadius,
                float maxSupportRadius,
                ShapeDescriptor::debug::SCExecutionTimes* executionTimes = nullptr);
    }
}