#pragma once

#include "spinImage/gpu/types/Mesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace debug {
        struct QSISearchRunInfo {
            double totalExecutionTimeSeconds;
            double searchExecutionTimeSeconds;
        };
    }

    namespace gpu {
        array<QuasiSpinImageSearchResults> findQuasiSpinImagesInHaystack(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<unsigned int> computeQuasiSpinImageSearchResultRanks(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount,
                SpinImage::debug::QSISearchRunInfo* runInfo = nullptr);
    }
}