#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/QUICCImages.h>

namespace SpinImage {
    namespace debug {
        struct QUICCIRunInfo {
            double generationTimeSeconds;
            double totalExecutionTimeSeconds;
        };
    }

    namespace gpu {
        QUICCIImages generateQUICCImages(
                array<radialIntersectionCountImagePixelType> RICIDescriptors,
                SpinImage::debug::QUICCIRunInfo* runinfo = nullptr);
    }
}