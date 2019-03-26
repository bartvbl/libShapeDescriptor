#include "correlationComputations.h"
#include <catch2/catch.hpp>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <spinImage/cpu/spinImageSearcher.h>
#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <spinImage/utilities/copy/hostDescriptorsToDevice.h>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include <iostream>
#include <spinImage/utilities/dumpers/spinImageDumper.h>

template<typename pixelType>
array<pixelType> generateEmptyImages(size_t imageCount) {
    pixelType* image = new pixelType[imageCount * spinImageWidthPixels * spinImageWidthPixels];
    array<pixelType> images;
    images.content = image;
    images.length = imageCount * spinImageWidthPixels * spinImageWidthPixels;

    return images;
}

template<typename pixelType>
array<pixelType> generateRepeatingTemplateImage(
        pixelType patternPart0,
        pixelType patternPart1,
        pixelType patternPart2,
        pixelType patternPart3,
        pixelType patternPart4,
        pixelType patternPart5,
        pixelType patternPart6,
        pixelType patternPart7) {

    array<pixelType> images = generateEmptyImages<pixelType>(1);

    for(size_t index = 0; index < spinImageWidthPixels * spinImageWidthPixels; index += 8) {
        images.content[index + 0] = patternPart0;
        images.content[index + 1] = patternPart1;
        images.content[index + 2] = patternPart2;
        images.content[index + 3] = patternPart3;
        images.content[index + 4] = patternPart4;
        images.content[index + 5] = patternPart5;
        images.content[index + 6] = patternPart6;
        images.content[index + 7] = patternPart7;
    }

    return images;
}

template<typename pixelType>
array<pixelType> generateKnownImageSequence(const int imageCount, const int pixelsPerImage) {
    array<pixelType> imageSequence = generateEmptyImages<pixelType>(imageCount);

    for(int image = 0; image < imageCount; image++) {
        for(int highIndex = 0; highIndex < image; highIndex++) {
            imageSequence.content[image * pixelsPerImage + highIndex] = 1;
        }
        for(int lowIndex = image; lowIndex < pixelsPerImage; lowIndex++) {
            imageSequence.content[image * pixelsPerImage + lowIndex] = 0;
        }
    }

    imageSequence.length = imageCount;
    return imageSequence;
}


TEST_CASE("Basic correlation computation", "[correlation]") {

    SECTION("Equivalent images (spin image)") {
        array<spinImagePixelType> constantImage =
                generateRepeatingTemplateImage<spinImagePixelType>(0, 1, 0, 1, 0, 1, 0, 1);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(constantImage.content, constantImage.content, 0,
                                                                        0);

        delete[] constantImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Equivalent images (quasi spin image)") {
        array<quasiSpinImagePixelType> constantImage =
                generateRepeatingTemplateImage<quasiSpinImagePixelType>(0, 1, 0, 1, 0, 1, 0, 1);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(constantImage.content, constantImage.content, 0,
                                                                        0);

        delete[] constantImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Opposite images (spin image)") {
        array<spinImagePixelType> positiveImage = generateEmptyImages<spinImagePixelType>(1);
        array<spinImagePixelType> negativeImage = generateEmptyImages<spinImagePixelType>(1);

        for (int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            positiveImage.content[i] = float(i);
            negativeImage.content[i] = float(i) * -1;
        }

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);
        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == -1);
    }

    SECTION("Opposite images (quasi spin image)") {
        array<quasiSpinImagePixelType> positiveImage = generateEmptyImages<quasiSpinImagePixelType>(1);
        array<quasiSpinImagePixelType> negativeImage = generateEmptyImages<quasiSpinImagePixelType>(1);

        for (int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            positiveImage.content[i] = unsigned(i);
            negativeImage.content[i] = unsigned(spinImageWidthPixels * spinImageWidthPixels - i);
        }

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);
        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == -1);
    }

    SECTION("Equivalent constant images (spin image)") {
        array<spinImagePixelType> positiveImage = generateRepeatingTemplateImage<spinImagePixelType>(5, 5, 5, 5, 5, 5,
                                                                                                     5, 5);
        array<spinImagePixelType> negativeImage = generateRepeatingTemplateImage<spinImagePixelType>(5, 5, 5, 5, 5, 5,
                                                                                                     5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);
        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Equivalent constant images (quasi spin image)") {
        array<quasiSpinImagePixelType> positiveImage = generateRepeatingTemplateImage<quasiSpinImagePixelType>(5, 5, 5,
                                                                                                               5, 5, 5,
                                                                                                               5, 5);
        array<quasiSpinImagePixelType> negativeImage = generateRepeatingTemplateImage<quasiSpinImagePixelType>(5, 5, 5,
                                                                                                               5, 5, 5,
                                                                                                               5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);
        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Different constant images (spin image)") {
        array<spinImagePixelType> positiveImage = generateRepeatingTemplateImage<spinImagePixelType>(
                2, 2, 2,
                2, 2, 2,
                2, 2);
        array<spinImagePixelType> negativeImage = generateRepeatingTemplateImage<spinImagePixelType>(
                5, 5, 5,
                5, 5, 5,
                5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0, 0);

        float otherCorrelation = SpinImage::cpu::computeImagePairCorrelation(negativeImage.content, positiveImage.content, 0, 0);

        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 0.4f);
        REQUIRE(otherCorrelation == 0.4f);
    }

    SECTION("Different constant images (quasi spin image)") {
        array<quasiSpinImagePixelType> positiveImage = generateRepeatingTemplateImage<quasiSpinImagePixelType>(
                2, 2, 2,
                2, 2, 2,
                2, 2);
        array<quasiSpinImagePixelType> negativeImage = generateRepeatingTemplateImage<quasiSpinImagePixelType>(
                5, 5, 5,
                5, 5, 5,
                5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0, 0);

        float otherCorrelation = SpinImage::cpu::computeImagePairCorrelation(negativeImage.content, positiveImage.content, 0, 0);

        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 0.4f);
        REQUIRE(otherCorrelation == 0.4f);
    }
}

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;
const float correlationThreshold = 0.000001f;

TEST_CASE("Ranking of search results on CPU", "[correlation]") {

    SECTION("Ranking by generating complete result set") {

        array<spinImagePixelType> imageSequence = generateKnownImageSequence<spinImagePixelType>(imageCount,
                                                                                                 pixelsPerImage);

        std::vector<std::vector<DescriptorSearchResult>> resultsCPU = SpinImage::cpu::findDescriptorsInHaystack(
                imageSequence, imageCount, imageSequence, imageCount);

        for (int i = 0; i < imageCount; i++) {
            float pairCorrelation = SpinImage::cpu::computeImagePairCorrelation(imageSequence.content,
                                                                                imageSequence.content, i, i);

            // We'll allow for some rounding errors here.
            REQUIRE(std::abs(pairCorrelation - 1.0f) < correlationThreshold);

            // Allow for shared first places
            int resultIndex = 0;
            while (std::abs(resultsCPU.at(i).at(resultIndex).correlation - 1.0f) < correlationThreshold) {
                if (resultsCPU.at(i).at(resultIndex).imageIndex == i) {
                    break;
                }
                resultIndex++;
            }

            REQUIRE(std::abs(resultsCPU.at(i).at(resultIndex).correlation - 1.0f) < correlationThreshold);
            REQUIRE(resultsCPU.at(i).at(resultIndex).imageIndex == i);
        }

        delete[] imageSequence.content;
    }
}

TEST_CASE("Ranking of search results on GPU") {

    SpinImage::utilities::createCUDAContext();

    array<spinImagePixelType> imageSequence = generateKnownImageSequence<spinImagePixelType>(imageCount, pixelsPerImage);

    array<spinImagePixelType> device_haystackImages = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

    SECTION("Ranking by generating search results on GPU") {
        array<ImageSearchResults> searchResults = SpinImage::gpu::findDescriptorsInHaystack(device_haystackImages, imageCount, device_haystackImages, imageCount);

        SECTION("Equivalent images are the top search results") {
            for (int image = 0; image < imageCount; image++) {
                // Allow for shared first places
                int resultIndex = 0;
                while (std::abs(searchResults.content[image].resultScores[resultIndex] - 1.0f) < correlationThreshold) {
                    if (searchResults.content[image].resultIndices[resultIndex] == image) {
                        break;
                    }
                    resultIndex++;
                }

                REQUIRE(std::abs(searchResults.content[image].resultScores[resultIndex] - 1.0f) < correlationThreshold);
                REQUIRE(searchResults.content[image].resultIndices[resultIndex] == image);
            }
        }

        SECTION("Scores are properly sorted") {
            for(int image = 0; image < imageCount; image++) {
                for (int i = 0; i < SEARCH_RESULT_COUNT - 1; i++) {
                    float firstImageCurrentScore = searchResults.content[image].resultScores[i];
                    float firstImageNextScore = searchResults.content[image].resultScores[i + 1];

                    REQUIRE(firstImageCurrentScore >= firstImageNextScore);
                }
            }
        }
    }

    SECTION("Ranking by computing rank indices") {

        array<unsigned int> results = SpinImage::gpu::computeSearchResultRanks(device_haystackImages, imageCount, device_haystackImages, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        array<spinImagePixelType> device_haystackImages_reversed = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

        array<unsigned int> results = SpinImage::gpu::computeSearchResultRanks(device_haystackImages_reversed, imageCount, device_haystackImages_reversed, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}