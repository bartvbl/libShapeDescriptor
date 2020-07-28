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
        SpinImage::array<unsigned int> compute3DSCSearchResultRanks(
                SpinImage::array<SpinImage::gpu::ShapeContextDescriptor> device_needleDescriptors,
                size_t needleDescriptorSampleCount,
                SpinImage::array<SpinImage::gpu::ShapeContextDescriptor> device_haystackDescriptors,
                size_t haystackDescriptorSampleCount,
                SpinImage::debug::SCSearchExecutionTimes* executionTimes = nullptr);
    }
}