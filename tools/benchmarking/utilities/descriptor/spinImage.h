#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/spinImageGenerator.h>
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
            ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> generateSpinImageDescriptor(
                ShapeDescriptor::cpu::Mesh mesh,
                std::string hardware,
                float supportRadius,
                float supportAngleDegrees,
                size_t sampleCount,
                size_t randomSeed,
                std::chrono::duration<double> &elapsedTime);
        }
    }
}