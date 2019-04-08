#pragma once

#include "spinImage/gpu/types/DeviceMesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace debug {
        struct SISearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
            double averagingExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<SpinImageSearchResults> findSpinImagesInHaystack(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<unsigned int> computeSpinImageSearchResultRanks(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::SISearchRunInfo* runInfo = nullptr);
    }
}