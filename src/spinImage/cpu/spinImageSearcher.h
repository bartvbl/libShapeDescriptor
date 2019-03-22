#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <vector>

struct DescriptorSearchResult {
    float correlation;
    size_t imageIndex;
};

namespace SpinImage {
    namespace cpu {
        std::vector<std::vector<DescriptorSearchResult>> findDescriptorsInHaystack(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        std::vector<std::vector<DescriptorSearchResult>> findDescriptorsInHaystack(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        float computeImagePairCorrelation(quasiSpinImagePixelType* descriptors,
                                          quasiSpinImagePixelType* otherDescriptors,
                                          size_t spinImageIndex,
                                          size_t otherImageIndex);

        float computeImagePairCorrelation(spinImagePixelType* descriptors,
                                          spinImagePixelType* otherDescriptors,
                                          size_t spinImageIndex,
                                          size_t otherImageIndex);
    }
}
