#pragma once

#include "types/Mesh.h"
#include "shapeDescriptor/common/types/OrientedPoint.h"
#include "types/array.h"
#include "shapeDescriptor/common/types/methods/RICIDescriptor.h"

namespace ShapeDescriptor {
    namespace cpu {
        struct RICIExecutionTimes {
            double generationTimeSeconds;
            double totalExecutionTimeSeconds;
        };

        ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> generateRadialIntersectionCountImages(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::cpu::RICIExecutionTimes* executionTimes = nullptr);
    }
}