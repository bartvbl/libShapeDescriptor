#pragma once

#include "shapeSearch/gpu/types/DeviceMesh.h"
#include "shapeSearch/common/types/array.h"
#include "shapeSearch/libraryBuildSettings.h"

struct ImageSearchResults {
    size_t resultIndices[SEARCH_RESULT_COUNT];
    float resultScores[SEARCH_RESULT_COUNT];
};

array<ImageSearchResults> findDescriptorsInHaystack(
        array<classicSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<classicSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount);

array<ImageSearchResults> findDescriptorsInHaystack(
        array<newSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<newSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount);



array<size_t> computeSearchResultRanks(
        array<classicSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<classicSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount);

array<size_t> computeSearchResultRanks(
        array<newSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<newSpinImagePixelType> device_haystackDescriptors,
        size_t haystackImageCount);