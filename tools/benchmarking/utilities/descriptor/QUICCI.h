#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/quickIntersectionCountImageGenerator.h>
#include <vector>
#include <variant>
#include <map>
#include <ctime>
#include <chrono>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> generateQUICCIDescriptor(
                ShapeDescriptor::cpu::Mesh mesh,
                std::string hardware,
                float supportRadius,
                std::chrono::duration<double> &elapsedTime);
        }
    }
}