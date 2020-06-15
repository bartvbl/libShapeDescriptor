#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/fastPointFeatureHistogramGenerator.cuh>

namespace SpinImage {
    namespace debug {
        struct FPFHSearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<unsigned int> computeFPFHSearchResultRanks(
                SpinImage::gpu::FPFHHistograms device_needleDescriptors,
                size_t needleDescriptorCount,
                SpinImage::gpu::FPFHHistograms device_haystackDescriptors,
                size_t haystackDescriptorCount,
                SpinImage::debug::FPFHSearchRunInfo* runInfo = nullptr);
    }
}