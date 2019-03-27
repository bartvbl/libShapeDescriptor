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

    float correlation;

    if(squaredSumX != 0 && squaredSumY != 0)
    {
        // In the usual case, we have non-constant images, and we compute the Pearson correlation
        // coefficient without any issues.
        correlation = multiplicativeSum / (squaredSumX * squaredSumY);
    } else if(squaredSumX == 0 && squaredSumY != 0) {


    } else if(squaredSumX != 0 && squaredSumY == 0) {

    } else if(squaredSumX == 0 && squaredSumY == 0 && averageX == averageY) {
        // If both sums are 0, both sequences must be constant
        // If any pair of pixels has the same value, by extension both images must be identical
        // Therefore, even though correlation is not defined at constant sequences,
        // the correlation value should be 1.
        correlation = 1;
    } else {
        // In case both images are constant, but have different values,
        // we define the correlation to be the fraction of their pixel values
        correlation = std::min(averageX, averageY) / std::abs(std::max(averageX, averageY));
    }

    return correlation;
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

template<typename pixelType>
std::vector<std::vector<DescriptorSearchResult>> computeCorrelations(
        array<pixelType> needleDescriptors,
        size_t needleImageCount,
        array<pixelType> haystackDescriptors,
        size_t haystackImageCount) {

    std::vector<std::vector<DescriptorSearchResult>> searchResults;
    searchResults.resize(needleImageCount);

    float* needleImageAverages = new float[needleImageCount];
    float* haystackImageAverages = new float[haystackImageCount];

    for(size_t i = 0; i < needleImageCount; i++) {
        needleImageAverages[i] = computeAveragePixelValue<pixelType>(needleDescriptors.content, i);
    }

    for(size_t i = 0; i < needleImageCount; i++) {
        haystackImageAverages[i] = computeAveragePixelValue<pixelType>(haystackDescriptors.content, i);
    }

#pragma omp parallel for
    for(size_t image = 0; image < needleImageCount; image++) {
        std::vector<DescriptorSearchResult> imageResults;
        float needleAverage = needleImageAverages[image];

        for(size_t haystackImage = 0; haystackImage < haystackImageCount; haystackImage++) {
            float haystackAverage = haystackImageAverages[haystackImage];

            float correlation = computePairCorrelation<pixelType>(needleDescriptors.content,
                                              haystackDescriptors.content, image, haystackImage, needleAverage, haystackAverage);

            DescriptorSearchResult entry;
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
    float averageX = computeAveragePixelValue<quasiSpinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeAveragePixelValue<quasiSpinImagePixelType>(otherDescriptors, otherImageIndex);
    return computePairCorrelation<quasiSpinImagePixelType>(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}

float SpinImage::cpu::computeImagePairCorrelation(spinImagePixelType* descriptors,
                                  spinImagePixelType* otherDescriptors,
                                  size_t spinImageIndex,
                                  size_t otherImageIndex) {
    float averageX = computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
    float averageY = computeAveragePixelValue<spinImagePixelType>(otherDescriptors, otherImageIndex);
    return computePairCorrelation<spinImagePixelType>(descriptors, otherDescriptors, spinImageIndex, otherImageIndex, averageX, averageY);
}

float SpinImage::cpu::computeImageAverage(spinImagePixelType* descriptors, size_t spinImageIndex) {
    return computeAveragePixelValue<spinImagePixelType>(descriptors, spinImageIndex);
}

float SpinImage::cpu::computeImageAverage(quasiSpinImagePixelType* descriptors, size_t spinImageIndex) {
    return computeAveragePixelValue<quasiSpinImagePixelType>(descriptors, spinImageIndex);
}