#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/fastPointFeatureHistogramGenerator.cuh>

namespace SpinImage {
    namespace debug {
        struct FPFHSearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<unsigned int> computeFPFHSearchResultRanks(
                SpinImage::array<SpinImage::gpu::FPFHDescriptor> device_needleDescriptors,
                SpinImage::array<SpinImage::gpu::FPFHDescriptor> device_haystackDescriptors,
                SpinImage::debug::FPFHSearchExecutionTimes* executionTimes = nullptr);
    }
}