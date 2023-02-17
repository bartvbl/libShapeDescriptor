#pragma once
#include <shapeDescriptor/cpu/radialIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
#include <vector>
#include <variant>
#include <shapeDescriptor/cpu/types/array.h>

namespace Benchmarking
{
    namespace utilities
    {
        namespace distance
        {
            double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata);
            double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata);
            double cosineSimilarityBetweenTwoDescriptors(ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsOne, ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> descriptorsTwo, std::vector<std::variant<int, std::string>> metadata);
        }
    }
}