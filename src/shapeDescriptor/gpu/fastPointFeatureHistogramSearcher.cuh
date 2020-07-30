#pragma once
#include <shapeDescriptor/gpu/fastPointFeatureHistogramGenerator.cuh>

namespace SpinImage {
    namespace debug {
        struct FPFHSearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::cpu::array<unsigned int> computeFPFHSearchResultRanks(
                SpinImage::gpu::array<SpinImage::gpu::FPFHDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::FPFHDescriptor> device_haystackDescriptors,
                SpinImage::debug::FPFHSearchExecutionTimes* executionTimes = nullptr);
    }
}