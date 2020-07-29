#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/methods/3DSCDescriptor.h>

namespace SpinImage {
    namespace debug {
        struct SCSearchExecutionTimes {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        SpinImage::cpu::array<unsigned int> compute3DSCSearchResultRanks(
                SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> device_needleDescriptors,
                size_t needleDescriptorSampleCount,
                SpinImage::gpu::array<SpinImage::gpu::ShapeContextDescriptor> device_haystackDescriptors,
                size_t haystackDescriptorSampleCount,
                SpinImage::debug::SCSearchExecutionTimes* executionTimes = nullptr);
    }
}