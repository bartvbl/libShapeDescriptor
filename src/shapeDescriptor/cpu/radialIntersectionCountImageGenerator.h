#pragma once

#include <shapeDescriptor/cpu/types/OrientedPoint.h>
#include "types/Mesh.h"
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
                ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint> descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::cpu::RICIExecutionTimes* executionTimes = nullptr);
    }
}