#pragma once

#include "shapeDescriptor/gpu/types/ImageSearchResults.h"
#include <shapeDescriptor/common/types/methods/SpinImageDescriptor.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>

namespace ShapeDescriptor {
    namespace debug {
        struct SISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
            double averagingExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<float>> findSpinImagesInHaystack(
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackDescriptors);

        ShapeDescriptor::cpu::array<unsigned int> computeSpinImageSearchResultRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackDescriptors,
                ShapeDescriptor::debug::SISearchExecutionTimes* executionTimes = nullptr);
    }
}