#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/gpu/types/Mesh.h>
#include <shapeDescriptor/gpu/3dShapeContextGenerator.cuh>
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
            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> generate3DShapeContextDescriptor(
                ShapeDescriptor::cpu::Mesh mesh,
                std::string hardware,
                size_t sampleCount,
                size_t randomSeed,
                float pointDensityRadius,
                float minSupportRadius,
                float maxSupportRadius,
                std::chrono::duration<double> &elapsedTime);
        }
    }
}