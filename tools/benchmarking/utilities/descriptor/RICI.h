#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <vector>
#include <map>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>> similarityBetweenTwoObjectsWithRICI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata);
        }
    }
}