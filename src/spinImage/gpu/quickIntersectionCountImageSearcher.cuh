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
        SpinImage::array<unsigned int> computeQUICCImageSearchResultRanks(
                SpinImage::gpu::QUICCIImages device_needleDescriptors,
                size_t needleImageCount,
                SpinImage::gpu::QUICCIImages device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::QUICCISearchRunInfo* runInfo = nullptr);

        struct QUICCIDistances {
            unsigned int clutterResistantDistance = 0;
            unsigned int hammingDistance = 0;
            float weightedHammingDistance = 0;
            unsigned int pixelCountDistance = 0;
        };

        SpinImage::array<SpinImage::gpu::QUICCIDistances> computeQUICCIElementWiseDistances(
                SpinImage::gpu::QUICCIImages device_descriptors,
                SpinImage::gpu::QUICCIImages device_correspondingDescriptors,
                size_t descriptorCount);
    }
}