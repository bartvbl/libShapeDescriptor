#pragma once

namespace SpinImage {
    namespace debug {
        struct QUICCISearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<RadialIntersectionCountImageSearchResults> findQUICCImagesInHaystack(
                array<unsigned int> device_needleDescriptors,
                size_t needleImageCount,
                array<unsigned int> device_haystackDescriptors,
                size_t haystackImageCount);

        array<unsigned int> computeQUICCImageSearchResultRanks(
                array<unsigned int> device_needleDescriptors,
                size_t needleImageCount,
                array<unsigned int> device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::QUICCISearchRunInfo* runInfo = nullptr);
    }
}