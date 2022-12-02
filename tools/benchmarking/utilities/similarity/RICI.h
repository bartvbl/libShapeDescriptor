#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>

namespace Benchmarking
{
    namespace utilities
    {
        namespace similarity
        {
            double similarityBetweenTwoObjectsWithRICI(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, double (*distanceAlgorithm)(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor>));
        }
    }
}