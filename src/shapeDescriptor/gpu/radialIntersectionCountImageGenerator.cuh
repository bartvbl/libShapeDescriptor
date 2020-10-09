#pragma once

#include "types/Mesh.h"
#include "shapeDescriptor/common/types/OrientedPoint.h"
#include "types/array.h"
#include "shapeDescriptor/common/types/methods/RICIDescriptor.h"

namespace ShapeDescriptor {
    namespace debug {
        struct RICIExecutionTimes {
            double generationTimeSeconds;
            double meshScaleTimeSeconds;
            double redistributionTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> generateRadialIntersectionCountImages(
                ShapeDescriptor::gpu::Mesh device_mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::debug::RICIExecutionTimes* executionTimes = nullptr);
    }
}