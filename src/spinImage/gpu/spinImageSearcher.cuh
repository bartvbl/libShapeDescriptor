#pragma once

#include "spinImage/gpu/types/Mesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"
#include <spinImage/gpu/types/methods/SpinImageDescriptor.h>

namespace SpinImage {
    namespace debug {
        struct SISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
            double averagingExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::array<SpinImageSearchResults> findSpinImagesInHaystack(
                SpinImage::array<SpinImage::gpu::SpinImageDescriptor> device_needleDescriptors,
                SpinImage::array<SpinImage::gpu::SpinImageDescriptor> device_haystackDescriptors);

        SpinImage::array<unsigned int> computeSpinImageSearchResultRanks(
                SpinImage::array<SpinImage::gpu::SpinImageDescriptor> device_needleDescriptors,
                SpinImage::array<SpinImage::gpu::SpinImageDescriptor> device_haystackDescriptors,
                SpinImage::debug::SISearchExecutionTimes* executionTimes = nullptr);
    }
}