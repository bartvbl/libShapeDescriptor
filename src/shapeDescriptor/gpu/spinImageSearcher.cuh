#pragma once

#include "shapeDescriptor/gpu/types/ImageSearchResults.h"
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace SpinImage {
    namespace debug {
        struct SISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
            double averagingExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::cpu::array<SpinImageSearchResults> findSpinImagesInHaystack(
                SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_haystackDescriptors);

        SpinImage::cpu::array<unsigned int> computeSpinImageSearchResultRanks(
                SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_haystackDescriptors,
                SpinImage::debug::SISearchExecutionTimes* executionTimes = nullptr);
    }
}