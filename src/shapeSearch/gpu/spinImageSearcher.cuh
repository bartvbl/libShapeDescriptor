#pragma once

#include "shapeSearch/gpu/types/DeviceMesh.h"
#include "shapeSearch/common/types/array.h"
#include "shapeSearch/libraryBuildSettings.h"

struct ImageSearchResults {
    size_t resultIndices[SEARCH_RESULT_COUNT];
    float resultScores[SEARCH_RESULT_COUNT];
};

namespace SpinImage {
    namespace gpu {
        array<ImageSearchResults> findDescriptorsInHaystack(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<ImageSearchResults> findDescriptorsInHaystack(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);


        array<size_t> computeSearchResultRanks(
                array<spinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<spinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);

        array<size_t> computeSearchResultRanks(
                array<quasiSpinImagePixelType> device_needleDescriptors,
                size_t needleImageCount,
                array<quasiSpinImagePixelType> device_haystackDescriptors,
                size_t haystackImageCount);
    }
}