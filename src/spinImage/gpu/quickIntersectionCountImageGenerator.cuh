#pragma once

#include <spinImage/common/types/array.h>

namespace SpinImage {
    namespace debug {
        struct QUICCIRunInfo {
            double generationTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<unsigned int> generateQUICCImages(
                array<radialIntersectionCountImagePixelType> RICIDescriptors,
                SpinImage::debug::QUICCIRunInfo* runinfo = nullptr);
    }
}