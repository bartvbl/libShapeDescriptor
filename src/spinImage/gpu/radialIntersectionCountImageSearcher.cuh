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
                array<SpinImage::gpu::RICIDescriptor> device_needleDescriptors,
                array<SpinImage::gpu::RICIDescriptor> device_haystackDescriptors);

        SpinImage::array<unsigned int> computeRadialIntersectionCountImageSearchResultRanks(
                SpinImage::array<SpinImage::gpu::RICIDescriptor> device_needleDescriptors,
                SpinImage::array<SpinImage::gpu::RICIDescriptor> device_haystackDescriptors,
                SpinImage::debug::RICISearchRunInfo* runInfo = nullptr);
    }
}