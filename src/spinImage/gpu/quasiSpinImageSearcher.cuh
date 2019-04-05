#pragma once

#include "spinImage/gpu/types/DeviceMesh.h"
#include "spinImage/common/types/array.h"
#include "spinImage/libraryBuildSettings.h"
#include "spinImage/gpu/types/ImageSearchResults.h"

namespace SpinImage {
    namespace gpu {
        array<unsigned int> computeSearchResultRanks(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);
    }
}