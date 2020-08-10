#pragma once

#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/common/types/methods/RICIDescriptor.h>
#include "shapeDescriptor/gpu/types/ImageSearchResults.h"

namespace ShapeDescriptor {
    namespace debug {
        struct RICISearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::cpu::array<RadialIntersectionCountImageSearchResults> findRadialIntersectionCountImagesInHaystack(
                ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackDescriptors);

        ShapeDescriptor::cpu::array<unsigned int> computeRadialIntersectionCountImageSearchResultRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackDescriptors,
                ShapeDescriptor::debug::RICISearchExecutionTimes* executionTimes = nullptr);
    }
}