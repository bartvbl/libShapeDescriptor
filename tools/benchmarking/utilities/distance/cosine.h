#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo);
        }
    }
}