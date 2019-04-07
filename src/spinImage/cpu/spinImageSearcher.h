#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <vector>

struct SpinImageSearchResult {
    float correlation;
    size_t imageIndex;
};

namespace SpinImage {
    namespace cpu {
        std::vector<std::vector<SpinImageSearchResult>> findSpinImagesInHaystack(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        float computeSpinImagePairCorrelation(spinImagePixelType* descriptors,
                                          spinImagePixelType* otherDescriptors,
                                          size_t spinImageIndex,
                                          size_t otherImageIndex);

        float computeImageAverage(spinImagePixelType* descriptors, size_t spinImageIndex);
    }
}
