#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace debug {
        struct SCSearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<unsigned int> compute3DSCSearchResultRanks(
                array<shapeContextBinType> device_needleDescriptors,
                size_t needleDescriptorCount,
                array<shapeContextBinType> device_haystackDescriptors,
                size_t haystackDescriptorCount,
                SpinImage::debug::SCSearchRunInfo* runInfo = nullptr);
    }
}