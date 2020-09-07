#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>

namespace ShapeDescriptor {
    namespace debug {
        struct SIExecutionTimes {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double generationTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> generateSpinImages(
                ShapeDescriptor::gpu::PointCloud device_pointCloud,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                float supportRadius,
                float supportAngleDegrees,
                ShapeDescriptor::debug::SIExecutionTimes* executionTimes = nullptr);
    }
}