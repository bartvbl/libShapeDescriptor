#pragma once
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/gpu/quickIntersectionCountImageGenerator.cuh>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace debug {
        struct QUICCISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        #if QUICCI_DISTANCE_FUNCTION == WEIGHTED_HAMMING_DISTANCE
                typedef float quicciDistanceType;
        #else
                typedef unsigned int quicciDistanceType;
        #endif

        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<quicciDistanceType>> findQUICCImagesInHaystack(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors);

        ShapeDescriptor::cpu::array<unsigned int> computeQUICCImageSearchResultRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_haystackDescriptors,
                ShapeDescriptor::debug::QUICCISearchExecutionTimes* executionTimes = nullptr);

        struct QUICCIDistances {
            unsigned int clutterResistantDistance = 0;
            unsigned int hammingDistance = 0;
            float weightedHammingDistance = 0;
            unsigned int needleImageBitCount = 0;
        };

        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::QUICCIDistances> computeQUICCIElementWiseDistances(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_descriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> device_correspondingDescriptors);
    }
}