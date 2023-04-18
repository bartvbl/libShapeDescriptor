#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/common/types/methods/3DSCDescriptor.h>

namespace ShapeDescriptor {
    namespace debug {
        struct SCSearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        ShapeDescriptor::cpu::array<unsigned int> compute3DSCSearchResultRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_needleDescriptors,
                size_t needleDescriptorSampleCount,
                ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_haystackDescriptors,
                size_t haystackDescriptorSampleCount,
                ShapeDescriptor::debug::SCSearchExecutionTimes* executionTimes = nullptr);

        ShapeDescriptor::cpu::array<float> compute3DSCElementWiseSquaredDistances(
                ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_descriptors,
                size_t descriptorSampleCount,
                ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> device_correspondingDescriptors,
                size_t correspondingDescriptorsSampleCount);
    }
}