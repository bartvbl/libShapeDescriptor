#include <cmath>
#include <algorithm>
#include <iostream>
#include "spinImageSearcher.h"

bool compareSearchResults(const SpinImageSearchResult &a, const SpinImageSearchResult &b)
{
    return a.correlation > b.correlation;
}


template<typename pixelType>
float computeAveragePixelValue(pixelType* descriptors, size_t spinImageIndex)
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


float computeSpinImagePairCorrelationCPU(
        spinImagePixelType* descriptors,
        spinImagePixelType* otherDescriptors,
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

            spinImagePixelType pixelValueX = descriptors[spinImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];
            spinImagePixelType pixelValueY = otherDescriptors[otherImageIndex * spinImageElementCount + (y * spinImageWidthPixels + x)];

            float deltaX = float(pixelValueX) - averageX;
            float deltaY = float(pixelValueY) - averageY;

            squaredSumX += deltaX * deltaX;
            squaredSumY += deltaY * deltaY;
            multiplicativeSum += deltaX * deltaY;
        }
    }

    squaredSumX = std::sqrt(squaredSumX);
    squaredSumY = std::sqrt(squaredSumY);

    // Assuming non-constant images
    // Will return NaN otherwise
    float correlation = multiplicativeSum / (squaredSumX * squaredSumY);

    return correlation;
}

std::vector<std::vector<SpinImageSearchResult>> computeCorrelations(
        array<spinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<spinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {

    std::vector<std::vector<SpinImageSearchResult>> searchResults;
    searchResults.resize(needleImageCount);

    float* needleImageAverages = new float[needleImageCount];
    float* haystackImageAverages = new float[haystackImageCount];

    for(size_t i = 0; i < needleImageCount; i++) {
        needleImageAverages[i] = computeAveragePixelValue<spinImagePixelType>(needleDescriptors.content, i);
    }

    for(size_t i = 0; i < needleImageCount; i++) {
        haystackImageAverages[i] = computeAveragePixelValue<spinImagePixelType>(haystackDescriptors.content, i);
    }

#pragma omp parallel for
    for(size_t image = 0; image < needleImageCount; image++) {
        std::vector<SpinImageSearchResult> imageResults;
        float needleAverage = needleImageAverages[image];

        for(size_t haystackImage = 0; haystackImage < haystackImageCount; haystackImage++) {
            float haystackAverage = haystackImageAverages[haystackImage];

            float correlation = computeSpinImagePairCorrelationCPU(
                    needleDescriptors.content,
                    haystackDescriptors.content,
                    image, haystackImage,
                    needleAverage, haystackAverage);

            SpinImageSearchResult entry;
            entry.correlation = correlation;
            entry.imageIndex = haystackImage;
            imageResults.push_back(entry);
        }

        std::sort(imageResults.begin(), imageResults.end(), compareSearchResults);
        searchResults.at(image) = imageResults;
    }

    delete[] needleImageAverages;
    delete[] haystackImageAverages;

    std::cout << "Analysed " << searchResults.size() << " images on the CPU." << std::endl;

    return searchResults;
}

std::vector<std::vector<SpinImageSearchResult>> SpinImage::cpu::findSpinImagesInHaystack(
        array<spinImagePixelType> needleDescriptors,
        size_t needleImageCount,
        array<spinImagePixelType> haystackDescriptors,
        size_t haystackImageCount) {
    return computeCorrelations(needleDescriptors, needleImageCount, haystackDescriptors, haystackImageCount);
}

float SpinImage::cpu::computeSpinImagePairCorrelation(spinImagePixelType* descriptors,
                                  spinImagePixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex) {
    float averageX = computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeAveragePixelValue<spinImagePixelType>(otherDescriptors, otherImageIndex);
    return computeSpinImagePairCorrelationCPU(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}

float SpinImage::cpu::computeImageAverage(spinImagePixelType* descriptors, size_t spinImageIndex) {
    return computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
}