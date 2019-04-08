#include <cmath>
#include <algorithm>
#include <iostream>
#include "quasiSpinImageSearcher.h"

bool compareSearchResults(const QuasiSpinImageSearchResult &a, const QuasiSpinImageSearchResult &b)
{
    // Including the image index as search criterion removes ambiguity when sorting,
    // and ensures we can compare the search results for equivalence in tests
    if(a.distance == b.distance) {
        return a.imageIndex < b.imageIndex;
    } else {
        return a.distance < b.distance;
    }
}

int compareQuasiSpinImagePairCPU(
        const quasiSpinImagePixelType* needleImages,
        const size_t needleImageIndex,
        const quasiSpinImagePixelType* haystackImages,
        const size_t haystackImageIndex) {

    int score = 0;
    const int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

    quasiSpinImagePixelType previousNeedlePixelValue = 0;
    quasiSpinImagePixelType previousHaystackPixelValue = 0;

    for(int pixel = 0; pixel < spinImageWidthPixels * spinImageWidthPixels; pixel ++) {
        quasiSpinImagePixelType currentNeedlePixelValue =
                needleImages[needleImageIndex * spinImageElementCount + pixel];
        quasiSpinImagePixelType currentHaystackPixelValue =
                haystackImages[haystackImageIndex * spinImageElementCount + pixel];

        if(pixel % spinImageWidthPixels > 0) {
            int needleDelta = int(currentNeedlePixelValue) - int(previousNeedlePixelValue);
            int haystackDelta = int(currentHaystackPixelValue) - int(previousHaystackPixelValue);

            if(needleDelta != 0) {
                score += (needleDelta - haystackDelta) * (needleDelta - haystackDelta);
            }
        }

        previousNeedlePixelValue = currentNeedlePixelValue;
        previousHaystackPixelValue = currentHaystackPixelValue;
    }

    return score;
}

std::vector<unsigned int> SpinImage::cpu::computeQuasiSpinImageRankIndices(
        quasiSpinImagePixelType* needleDescriptors,
        quasiSpinImagePixelType* haystackDescriptors,
        size_t imageCount) {
    std::vector<unsigned int> searchResults;
    searchResults.resize(imageCount);

#pragma omp parallel for
    for(size_t needleImageIndex = 0; needleImageIndex < imageCount; needleImageIndex++) {

        int referenceScore = compareQuasiSpinImagePairCPU(needleDescriptors, needleImageIndex, haystackDescriptors, needleImageIndex);

        if (referenceScore == 0) {
            continue;
        }

        unsigned int searchResultRank = 0;

        for (size_t haystackImageIndex = 0; haystackImageIndex < imageCount; haystackImageIndex++) {
            if (needleImageIndex == haystackImageIndex) {
                continue;
            }

            int pairScore = compareQuasiSpinImagePairCPU(needleDescriptors, needleImageIndex, haystackDescriptors, haystackImageIndex);

            if (pairScore < referenceScore) {
                searchResultRank++;
            }
        }

        searchResults.at(needleImageIndex) = searchResultRank;
    }

    return searchResults;
}

std::vector<std::vector<QuasiSpinImageSearchResult>> computeQuasiSpinImageRankings(
        quasiSpinImagePixelType* needleDescriptors,
        size_t needleImageCount,
        quasiSpinImagePixelType* haystackDescriptors,
        size_t haystackImageCount) {

    std::vector<std::vector<QuasiSpinImageSearchResult>> searchResults;
    searchResults.resize(needleImageCount);

#pragma omp parallel for
    for(size_t image = 0; image < needleImageCount; image++) {
        std::vector<QuasiSpinImageSearchResult> imageResults;

        for(size_t haystackImage = 0; haystackImage < haystackImageCount; haystackImage++) {

            int score = compareQuasiSpinImagePairCPU(
                    needleDescriptors,
                    image,
                    haystackDescriptors,
                    haystackImage);

            QuasiSpinImageSearchResult entry;
            entry.distance = score;
            entry.imageIndex = haystackImage;
            imageResults.push_back(entry);
        }

        std::sort(imageResults.begin(), imageResults.end(), compareSearchResults);
        searchResults.at(image) = imageResults;
    }

    std::cout << "Analysed " << searchResults.size() << " images on the CPU." << std::endl;

    return searchResults;
}

std::vector<std::vector<QuasiSpinImageSearchResult>> SpinImage::cpu::findQuasiSpinImagesInHaystack(
        array<quasiSpinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<quasiSpinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeQuasiSpinImageRankings(needleDescriptors.content, needleImageCount, haystackDescriptors.content, haystackImageCount);
}

int SpinImage::cpu::computeQuasiSpinImagePairDistance(
        quasiSpinImagePixelType* descriptors,
        quasiSpinImagePixelType* otherDescriptors,
        size_t spinImageIndex,
        size_t otherImageIndex) {
    return compareQuasiSpinImagePairCPU(descriptors, spinImageIndex, otherDescriptors, otherImageIndex);
}