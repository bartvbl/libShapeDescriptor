#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/PointCloud.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>

namespace ShapeDescriptor {
    namespace cpu {
        struct SIExecutionTimes {
            double totalExecutionTimeSeconds;
            double initialisationTimeSeconds;
            double generationTimeSeconds;
        };
        
        ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> generateSpinImages(
                ShapeDescriptor::cpu::PointCloud pointCloud,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                float supportRadius,
                float supportAngleDegrees,
                ShapeDescriptor::cpu::SIExecutionTimes* executionTimes = nullptr);
    }
}