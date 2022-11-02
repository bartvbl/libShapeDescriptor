#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/QUICCIDescriptor.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>

namespace ShapeDescriptor {
    namespace cpu {
        struct QUICCIExecutionTimes {
            double generationTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace cpu {
        ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> generateQUICCImages(
                Mesh device_mesh,
                ShapeDescriptor::cpu::array<OrientedPoint> device_descriptorOrigins,
                float supportRadius,
                ShapeDescriptor::cpu::QUICCIExecutionTimes* executionTimes = nullptr);
    }
}