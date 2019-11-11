#pragma once

#include "spinImage/gpu/types/Mesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace debug {
        struct RICISearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<RadialIntersectionCountImageSearchResults> findRadialIntersectionCountImagesInHaystack(
                array<radialIntersectionCountImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<radialIntersectionCountImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<unsigned int> computeRadialIntersectionCountImageSearchResultRanks(
                array<radialIntersectionCountImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<radialIntersectionCountImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::RICISearchRunInfo* runInfo = nullptr);
    }
}