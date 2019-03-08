#include "shapeSearch/gpu/types/DeviceMesh.h"
#include "shapeSearch/common/types/array.h"

struct ImageSearchResults {
    size_t resultIndices[32];
    float resultScores[32];
};

array<ImageSearchResults> findDescriptorsInHaystack(
        array<classicSpinImagePixelType > device_needleDescriptors,
        size_t needleImageCount,
        array<classicSpinImagePixelType > device_haystackDescriptors,
        size_t haystackImageCount);

array<ImageSearchResults> findDescriptorsInHaystack(
        array<newSpinImagePixelType> device_needleDescriptors,
        size_t needleImageCount,
        array<newSpinImagePixelType > device_haystackDescriptors,
        size_t haystackImageCount);