#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>
#include <shapeDescriptor/utilities/copy/mesh.h>
#include <vector>
#include <map>
#include <variant>
#include <ctime>
#include <chrono>

namespace Benchmarking
{
    namespace utilities
    {
        namespace descriptor
        {
            ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> generateFPFHDescriptor(
                ShapeDescriptor::cpu::Mesh mesh,
                std::string hardware,
                float supportRadius,
                size_t sampleCount,
                size_t randomSeed,
                std::chrono::duration<double> &elapsedTime);
        }
    }
}