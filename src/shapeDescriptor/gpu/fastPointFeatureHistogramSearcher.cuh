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
                ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_haystackDescriptors,
                ShapeDescriptor::debug::FPFHSearchExecutionTimes* executionTimes = nullptr);

        ShapeDescriptor::cpu::array<float> computeFPFHElementWiseEuclideanDistances(
                ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_descriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> device_correspondingDescriptors);
    }
}