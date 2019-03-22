#include <cmath>
#include <algorithm>
#include <iostream>
#include "spinImageSearcher.h"

bool compareSearchResults(const DescriptorSearchResult &a, const DescriptorSearchResult &b)
{
    return a.correlation > b.correlation;
}

template<typename pixelType>
float computePairCorrelation(pixelType* descriptors,
                                  pixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex,
                                  float averageX, float averageY) {
    float squaredSumX = 0;
    float squaredSumY = 0;
    float multiplicativeSum = 0;

    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels; x++)
        {
            const size_t spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

            pixelType pixelValueX = descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
            pixelType pixelValueY = otherDescriptors[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

            float deltaX = float(pixelValueX) - averageX;
            float deltaY = float(pixelValueY) - averageY;

            squaredSumX += deltaX * deltaX;
            squaredSumY += deltaY * deltaY;
            multiplicativeSum += deltaX * deltaY;
        }
    }

    squaredSumX = std::sqrt(squaredSumX);
    squaredSumY = std::sqrt(squaredSumY);

    float correlation = -1;

    if(squaredSumX != 0 || squaredSumY != 0)
    {
        // Avoiding zero divisions
        const float smallestNonZeroFactor = 0.0001;
        squaredSumX = std::max(squaredSumX, smallestNonZeroFactor);
        squaredSumY = std::max(squaredSumY, smallestNonZeroFactor);
        multiplicativeSum = std::max(multiplicativeSum, smallestNonZeroFactor * smallestNonZeroFactor);

        correlation = multiplicativeSum / (squaredSumX * squaredSumY);
    } else if(squaredSumX == 0 && squaredSumY == 0) {
        // If both sums are 0, both sequences must be identical
        correlation = 1;
    }

    return correlation;
}

template<typename pixelType>
float computeImageAverage(pixelType* descriptors, size_t spinImageIndex)
{
    const unsigned int spinImageElementCount = spinImageWidthPixels * spinImageWidthPixels;

    float sum = 0;

    for (int y = 0; y < spinImageWidthPixels; y++)
    {
        for (int x = 0; x < spinImageWidthPixels; x ++)
        {
            float pixelValue = float(descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)]);
            sum += pixelValue;
        }
    }

    return sum / float(spinImageElementCount);
}

template<typename pixelType>
float computeCorrelation(const array<pixelType> &needleDescriptors, const array<pixelType> &haystackDescriptors,
                          size_t needleImageIndex, size_t haystackImageIndex) {
    float needleAverage = computeImageAverage<pixelType>(needleDescriptors.content, needleImageIndex);
    float haystackAverage = computeImageAverage<pixelType>(haystackDescriptors.content, haystackImageIndex);

    return computePairCorrelation<pixelType>(needleDescriptors.content,
            haystackDescriptors.content, needleImageIndex, haystackImageIndex, needleAverage, haystackAverage);
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
            float correlation = computeCorrelation<pixelType>(needleDescriptors, haystackDescriptors, image, haystackImage);

            DescriptorSearchResult entry;
            entry.correlation = correlation;
            entry.imageIndex = haystackImage;
            imageResults.push_back(entry);
        }

        std::sort(imageResults.begin(), imageResults.end(), compareSearchResults);
        searchResults.at(image) = imageResults;
    }

    std::cout << "Analysed " << searchResults.size() << " images on the CPU." << std::endl;

    return searchResults;
}

std::vector<std::vector<DescriptorSearchResult>> SpinImage::cpu::findDescriptorsInHaystack(
        array<spinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<spinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations<spinImagePixelType>(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}

std::vector<std::vector<DescriptorSearchResult>> SpinImage::cpu::findDescriptorsInHaystack(
        array<quasiSpinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<quasiSpinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations<quasiSpinImagePixelType>(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}

float SpinImage::cpu::computeImagePairCorrelation(quasiSpinImagePixelType* descriptors,
                                  quasiSpinImagePixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex) {
    float averageX = computeImageAverage<quasiSpinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeImageAverage<quasiSpinImagePixelType>(otherDescriptors, otherImageIndex);
    return computePairCorrelation<quasiSpinImagePixelType>(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}

float SpinImage::cpu::computeImagePairCorrelation(spinImagePixelType* descriptors,
                                  spinImagePixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex) {
    float averageX = computeImageAverage<spinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeImageAverage<spinImagePixelType>(otherDescriptors, otherImageIndex);
    return computePairCorrelation<spinImagePixelType>(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}