#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <vector>
#include <variant>
#include <map>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            std::map<int, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor>> generateSpinImageDescriptors(ShapeDescriptor::cpu::Mesh meshOne, ShapeDescriptor::cpu::Mesh meshTwo, std::vector<std::variant<int, std::string>> metadata, std::string hardware);
        }
    }
}