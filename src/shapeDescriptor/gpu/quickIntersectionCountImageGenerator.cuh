#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/common/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>

namespace ShapeDescriptor {
    namespace debug {
        struct QUICCIExecutionTimes {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> generateQUICCImages(
                Mesh device_mesh,
                ShapeDescriptor::gpu::array<OrientedPoint> device_descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::debug::QUICCIExecutionTimes* executionTimes = nullptr);
    }
}