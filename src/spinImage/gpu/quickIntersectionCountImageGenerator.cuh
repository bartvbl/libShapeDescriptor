#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace debug {
        struct QUICCIRunInfo {
            double generationTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        struct QUICCIImages {
            unsigned int* horizontallyIncreasingImages;
            unsigned int* horizontallyDecreasingImages;
            size_t imageCount;
        };

        QUICCIImages generateQUICCImages(
                array<radialIntersectionCountImagePixelType> RICIDescriptors,
                SpinImage::debug::QUICCIRunInfo* runinfo = nullptr);
    }
}