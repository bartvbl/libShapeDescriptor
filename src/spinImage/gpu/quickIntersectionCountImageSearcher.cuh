#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/gpu/types/ImageSearchResults.h>
#include <spinImage/gpu/quickIntersectionCountImageGenerator.cuh>

namespace SpinImage {
    namespace debug {
        struct QUICCISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::array<unsigned int> computeQUICCImageSearchResultRanks(
                SpinImage::array<SpinImage::gpu::QUICCIDescriptor> device_needleDescriptors,
                SpinImage::array<SpinImage::gpu::QUICCIDescriptor> device_haystackDescriptors,
                SpinImage::debug::QUICCISearchExecutionTimes* executionTimes = nullptr);

        struct QUICCIDistances {
            unsigned int clutterResistantDistance = 0;
            unsigned int hammingDistance = 0;
            float weightedHammingDistance = 0;
            unsigned int needleImageBitCount = 0;
        };

        SpinImage::array<SpinImage::gpu::QUICCIDistances> computeQUICCIElementWiseDistances(
                SpinImage::array<SpinImage::gpu::QUICCIDescriptor> device_descriptors,
                SpinImage::array<SpinImage::gpu::QUICCIDescriptor> device_correspondingDescriptors);
    }
}