#include "spinImageSearcher.h"

template<typename pixelType>
float computeCorrelations(const array<pixelType> &needleDescriptors, const array<pixelType> &haystackDescriptors,
                          size_t image, size_t haystackImage) {

}

template<typename pixelType>
std::vector<std::vector<DescriptorSearchResult>> computeCorrelations(
        array<pixelType> needleDescriptors,
        size_t needleImageCount,
        array<pixelType> haystackDescriptors,
        size_t haystackImageCount) {

    std::vector<std::vector<DescriptorSearchResult>> searchResults;
    searchResults.resize(needleImageCount);

#pragma omp parallel for
    for(size_t image = 0; image < needleImageCount; image++) {
        std::vector<DescriptorSearchResult> imageResults;

        for(size_t haystackImage = 0; haystackImage < haystackImageCount; haystackImage++) {
            float correlation = computeCorrelations<pixelType>(needleDescriptors, haystackDescriptors, image, haystackImage);

            DescriptorSearchResult entry;
            entry.correlation = correlation;
            entry.imageIndex = haystackImage;
            imageResults.push_back(entry);
        }

        searchResults.push_back(imageResults);
    }

    return searchResults;
}

std::vector<std::vector<DescriptorSearchResult>> ShapeSearchCPU::findDescriptorsInHaystack(
        array<classicSpinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<classicSpinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations<classicSpinImagePixelType>(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}

std::vector<std::vector<DescriptorSearchResult>> ShapeSearchCPU::findDescriptorsInHaystack(
        array<newSpinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<newSpinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations<newSpinImagePixelType>(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}
