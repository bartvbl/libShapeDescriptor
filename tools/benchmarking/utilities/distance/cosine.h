#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <vector>

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata);
        }
    }
}