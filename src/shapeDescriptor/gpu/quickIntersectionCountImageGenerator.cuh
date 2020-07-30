#pragma once

#include <spinImage/gpu/types/array.h>
#include <spinImage/gpu/types/Mesh.h>
#include <spinImage/gpu/types/DeviceOrientedPoint.h>
#include <spinImage/gpu/types/methods/QUICCIDescriptor.h>

namespace SpinImage {
    namespace debug {
        struct QUICCIExecutionTimes {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> generateQUICCImages(
                Mesh device_mesh,
                SpinImage::gpu::array<DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                SpinImage::debug::QUICCIExecutionTimes* executionTimes = nullptr);
    }
}