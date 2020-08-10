#pragma once

#include "types/Mesh.h"
#include "types/DeviceOrientedPoint.h"
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
        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::RICIDescriptor> generateRadialIntersectionCountImages(
                ShapeDescriptor::gpu::Mesh device_mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::DeviceOrientedPoint> device_descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::debug::RICIExecutionTimes* executionTimes = nullptr);
    }
}