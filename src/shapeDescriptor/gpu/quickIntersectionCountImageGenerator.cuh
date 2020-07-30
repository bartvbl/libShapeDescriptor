#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/DeviceOrientedPoint.h>
#include <shapeDescriptor/gpu/types/methods/QUICCIDescriptor.h>

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