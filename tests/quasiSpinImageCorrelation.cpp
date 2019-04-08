#include "quasiSpinImageCorrelation.h"

#include <catch2/catch.hpp>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/cpu/spinImageSearcher.h>
#include <spinImage/cpu/quasiSpinImageSearcher.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <spinImage/utilities/copy/hostDescriptorsToDevice.h>
#include <spinImage/gpu/types/ImageSearchResults.h>
#include <spinImage/gpu/quasiSpinImageSearcher.cuh>
#include <iostream>
#include <spinImage/utilities/dumpers/searchResultDumper.h>
#include "utilities/spinImageGenerator.h"

TEST_CASE("Basic correlation computation (Quasi Spin Images)", "[correlation]") {

    const int repetitionsPerImage = (spinImageWidthPixels * spinImageWidthPixels) / 8;

    SECTION("Equivalent images") {
        array<quasiSpinImagePixelType> constantImage = generateRepeatingTemplateQuasiSpinImage(0, 1, 0, 1, 0, 1, 0, 1);

        int distance = SpinImage::cpu::computeQuasiSpinImagePairDistance(constantImage.content, constantImage.content, 0, 0);

        delete[] constantImage.content;
        REQUIRE(distance == 0);
    }

    SECTION("Clutter is ignored") {
        array<quasiSpinImagePixelType> needleImage =   generateRepeatingTemplateQuasiSpinImage(0, 0, 0, 1, 0, 0, 0, 0);
        array<quasiSpinImagePixelType> haystackImage = generateRepeatingTemplateQuasiSpinImage(0, 1, 0, 1, 0, 1, 0, 1);

        float distance = SpinImage::cpu::computeQuasiSpinImagePairDistance(needleImage.content, haystackImage.content, 0, 0);

        delete[] needleImage.content;
        delete[] haystackImage.content;

        REQUIRE(distance == 0);
    }

    SECTION("Difference is sum of squared distance") {
        array<quasiSpinImagePixelType> needleImage =   generateRepeatingTemplateQuasiSpinImage(0, 0, 0, 2, 0, 0, 0, 0);
        array<quasiSpinImagePixelType> haystackImage = generateRepeatingTemplateQuasiSpinImage(0, 1, 0, 4, 0, 1, 0, 1);

        int distance = SpinImage::cpu::computeQuasiSpinImagePairDistance(needleImage.content, haystackImage.content, 0, 0);

        delete[] needleImage.content;
        delete[] haystackImage.content;

        REQUIRE(distance == 2 * (2 * 2) * repetitionsPerImage);
    }

    SECTION("Equivalent constant images") {
        array<quasiSpinImagePixelType> positiveImage = generateRepeatingTemplateQuasiSpinImage(
                5, 5, 5, 5, 5, 5, 5, 5);
        array<quasiSpinImagePixelType> negativeImage = generateRepeatingTemplateQuasiSpinImage(
                5, 5, 5, 5, 5, 5, 5, 5);

        int distance = SpinImage::cpu::computeQuasiSpinImagePairDistance(positiveImage.content, negativeImage.content, 0, 0);

        delete[] positiveImage.content;
        delete[] negativeImage.content;

        REQUIRE(distance == 0);
    }
}

const float correlationThreshold = 0.00001f;


TEST_CASE("Ranking of Quasi Spin Images on the GPU") {

    SpinImage::utilities::createCUDAContext();

    array<quasiSpinImagePixelType> imageSequence = generateKnownQuasiSpinImageSequence(imageCount, pixelsPerImage);

    array<quasiSpinImagePixelType> device_haystackImages = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

    SECTION("Ranking by generating search results on GPU") {
        array<QuasiSpinImageSearchResults> searchResults = SpinImage::gpu::findQuasiSpinImagesInHaystack(device_haystackImages, imageCount, device_haystackImages, imageCount);

        SpinImage::dump::searchResults(searchResults, imageCount, "qsi_another_dump.txt");

        SECTION("Equivalent images are the top search results") {
            // First and last image are constant, which causes the pearson correlation to be undefined.
            // We therefore exclude them from the test.
            for (int image = 0; image < imageCount; image++) {
                // Allow for shared first places
                int resultIndex = 0;
                while (searchResults.content[image].resultScores[resultIndex] == 0) {
                    if (searchResults.content[image].resultIndices[resultIndex] == image) {
                        break;
                    }
                    resultIndex++;
                }
                std::cout << image << std::endl;
                REQUIRE(searchResults.content[image].resultScores[resultIndex] == 0);
                REQUIRE(searchResults.content[image].resultIndices[resultIndex] == image);
            }
        }

        SECTION("Scores are properly sorted") {
            for(int image = 0; image < imageCount; image++) {
                for (int i = 0; i < SEARCH_RESULT_COUNT - 1; i++) {
                    float firstImageCurrentScore = searchResults.content[image].resultScores[i];
                    float firstImageNextScore = searchResults.content[image].resultScores[i + 1];

                    REQUIRE(firstImageCurrentScore <= firstImageNextScore);
                }
            }
        }
    }

    SECTION("Ranking by computing rank indices") {

        array<unsigned int> results = SpinImage::gpu::computeQuasiSpinImageSearchResultRanks(device_haystackImages, imageCount, device_haystackImages, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        array<quasiSpinImagePixelType> device_haystackImages_reversed = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

        array<unsigned int> results = SpinImage::gpu::computeQuasiSpinImageSearchResultRanks(device_haystackImages_reversed, imageCount, device_haystackImages_reversed, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}