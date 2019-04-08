#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <vector>

struct QuasiSpinImageSearchResult {
    int distance;
    size_t imageIndex;
};

namespace SpinImage {
    namespace cpu {
        std::vector<std::vector<QuasiSpinImageSearchResult>> findQuasiSpinImagesInHaystack(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        int computeQuasiSpinImagePairDistance(
                quasiSpinImagePixelType* descriptors,
                quasiSpinImagePixelType* otherDescriptors,
                size_t spinImageIndex,
                size_t otherImageIndex);

        std::vector<unsigned int> computeQuasiSpinImageRankIndices(
                quasiSpinImagePixelType* needleDescriptors,
                quasiSpinImagePixelType* haystackDescriptors,
                size_t imageCount);
    }
}
