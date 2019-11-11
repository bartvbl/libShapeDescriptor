#include "radialIntersectionCountImageCorrelation.h"

#include <catch2/catch.hpp>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <spinImage/utilities/copy/hostDescriptorsToDevice.h>
#include <spinImage/gpu/types/ImageSearchResults.h>
#include <spinImage/gpu/radialIntersectionCountImageSearcher.cuh>
#include <iostream>
#include <spinImage/utilities/dumpers/searchResultDumper.h>
#include "utilities/spinImageGenerator.h"

const float correlationThreshold = 0.00001f;


TEST_CASE("Ranking of Quasi Spin Images on the GPU") {

    SpinImage::utilities::createCUDAContext();

    SpinImage::array<radialIntersectionCountImagePixelType> imageSequence = generateKnownQuasiSpinImageSequence(imageCount, pixelsPerImage);

    SpinImage::array<radialIntersectionCountImagePixelType> device_haystackImages = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

    SECTION("Ranking by generating search results on GPU") {
        SpinImage::array<SpinImage::gpu::RadialIntersectionCountImageSearchResults> searchResults = SpinImage::gpu::findRadialIntersectionCountImagesInHaystack(
                device_haystackImages, imageCount, device_haystackImages, imageCount);

        SpinImage::dump::searchResults(searchResults, imageCount, "rici_another_dump.txt");

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

        SpinImage::array<unsigned int> results = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                device_haystackImages, imageCount, device_haystackImages, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        SpinImage::array<radialIntersectionCountImagePixelType> device_haystackImages_reversed = SpinImage::copy::hostDescriptorsToDevice(imageSequence, imageCount);

        SpinImage::array<unsigned int> results = SpinImage::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                device_haystackImages_reversed, imageCount, device_haystackImages_reversed, imageCount);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}