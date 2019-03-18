#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/common/types/array.h>
#include <vector>

struct DescriptorSearchResult {
    float correlation;
    size_t imageIndex;
};

namespace ShapeSearchCPU {
    std::vector<std::vector<DescriptorSearchResult>> findDescriptorsInHaystack(
            array<classicSpinImagePixelType> device_needleDescriptors,
            size_t needleImageCount,
            array<classicSpinImagePixelType> device_haystackDescriptors,
            size_t haystackImageCount);

    std::vector<std::vector<DescriptorSearchResult>> findDescriptorsInHaystack(
            array<newSpinImagePixelType> device_needleDescriptors,
            size_t needleImageCount,
            array<newSpinImagePixelType> device_haystackDescriptors,
            size_t haystackImageCount);
}
