#pragma once

#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/common/types/methods/3DSCDescriptor.h>

namespace SpinImage {
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
        SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> generate3DSCDescriptors(
                SpinImage::gpu::PointCloud device_pointCloud,
                SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                float pointDensityRadius,
                float minSupportRadius,
                float maxSupportRadius,
                SpinImage::debug::SCExecutionTimes* executionTimes = nullptr);
    }
}