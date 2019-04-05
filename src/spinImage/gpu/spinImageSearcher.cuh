#pragma once

#include "spinImage/gpu/types/DeviceMesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace debug {
        struct SISearchRunInfo {
            double totalExecutionTime;
            double searchExecutionTime;
        };
    }

    namespace gpu {
        array<ImageSearchResults> findDescriptorsInHaystack(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<unsigned int> computeSearchResultRanks(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::SISearchRunInfo* runInfo = nullptr);
    }
}