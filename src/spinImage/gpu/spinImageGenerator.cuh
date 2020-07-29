#pragma once

#include <spinImage/gpu/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/methods/SpinImageDescriptor.h>

namespace SpinImage {
    namespace debug {
        struct SIExecutionTimes {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double generationTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> generateSpinImages(
                SpinImage::gpu::PointCloud device_pointCloud,
                SpinImage::gpu::array<SpinImage::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                float supportAngleDegrees,
                SpinImage::debug::SIExecutionTimes* executionTimes = nullptr);
    }
}