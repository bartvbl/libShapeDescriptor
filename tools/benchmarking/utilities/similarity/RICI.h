#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <vector>

namespace Benchmarking
{
    namespace utilities
    {
        namespace similarity
        {
            double similarityBetweenTwoObjectsWithRICI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, double (*distanceAlgorithm)(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, std::vector<std::variant<int, std::string>>), std::vector<std::variant<int, std::string>> metadata);
        }
    }
}