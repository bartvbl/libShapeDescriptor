#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/ImageSearchResults.h>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>

namespace SpinImage {
    namespace debug {
        struct QUICCISearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<unsigned int> computeQUICCImageSearchResultRanks(
                SpinImage::gpu::QUICCIImages device_needleDescriptors,
                size_t needleImageCount,
                SpinImage::gpu::QUICCIImages device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::QUICCISearchRunInfo* runInfo = nullptr);
    }
}