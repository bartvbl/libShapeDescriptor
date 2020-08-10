#pragma once
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>

namespace ShapeDescriptor {
    namespace debug {
        struct FPFHSearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::cpu::array<unsigned int> computeFPFHSearchResultRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::FPFHDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::FPFHDescriptor> device_haystackDescriptors,
                ShapeDescriptor::debug::FPFHSearchExecutionTimes* executionTimes = nullptr);
    }
}