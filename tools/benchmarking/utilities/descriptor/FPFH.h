#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <map>
#include <variant>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor>> generateFPFHDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware);
        }
    }
}