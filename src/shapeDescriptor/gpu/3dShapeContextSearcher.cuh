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
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::ShapeContextDescriptor> device_needleDescriptors,
                size_t needleDescriptorSampleCount,
                ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::ShapeContextDescriptor> device_haystackDescriptors,
                size_t haystackDescriptorSampleCount,
                ShapeDescriptor::debug::SCSearchExecutionTimes* executionTimes = nullptr);
    }
}