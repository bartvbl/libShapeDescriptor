#pragma once
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace SpinImage {
    namespace debug {
        struct QUICCISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::cpu::array<unsigned int> computeQUICCImageSearchResultRanks(
                SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> device_needleDescriptors,
                SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> device_haystackDescriptors,
                SpinImage::debug::QUICCISearchExecutionTimes* executionTimes = nullptr);

        struct QUICCIDistances {
            unsigned int clutterResistantDistance = 0;
            unsigned int hammingDistance = 0;
            float weightedHammingDistance = 0;
            unsigned int needleImageBitCount = 0;
        };

        SpinImage::cpu::array<SpinImage::gpu::QUICCIDistances> computeQUICCIElementWiseDistances(
                SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> device_descriptors,
                SpinImage::gpu::array<SpinImage::gpu::QUICCIDescriptor> device_correspondingDescriptors);
    }
}