#pragma once

#include "spinImage/gpu/types/Mesh.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"
#include <spinImage/gpu/types/methods/SpinImageDescriptor.h>
#include <spinImage/gpu/types/array.h>
#include <spinImage/cpu/types/array.h>

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