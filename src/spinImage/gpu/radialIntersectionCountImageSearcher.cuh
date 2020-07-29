#pragma once

#include "spinImage/gpu/types/Mesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace debug {
        struct RICISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::cpu::array<RadialIntersectionCountImageSearchResults> findRadialIntersectionCountImagesInHaystack(
                SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> device_haystackDescriptors);

        SpinImage::cpu::array<unsigned int> computeRadialIntersectionCountImageSearchResultRanks(
                SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::RICIDescriptor> device_haystackDescriptors,
                SpinImage::debug::RICISearchExecutionTimes* executionTimes = nullptr);
    }
}