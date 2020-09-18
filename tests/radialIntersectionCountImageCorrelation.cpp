#include <catch2/catch.hpp>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <shapeDescriptor/gpu/types/ImageSearchResults.h>
#include <shapeDescriptor/gpu/radialIntersectionCountImageSearcher.cuh>
#include <iostream>
#include <shapeDescriptor/utilities/dumpers/searchResultDumper.h>
#include <shapeDescriptor/utilities/copy/array.h>
#include "utilities/spinImageGenerator.h"

const float correlationThreshold = 0.00001f;


TEST_CASE("Ranking of Radial Intersection Count Images on the GPU") {

    ShapeDescriptor::utilities::createCUDAContext();

    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> imageSequence = generateKnownRadialIntersectionCountImageSequence(
            imageCount, pixelsPerImage);

    ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackImages = ShapeDescriptor::copy::hostArrayToDevice(imageSequence);

    SECTION("Ranking by generating search results on GPU") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SearchResults<unsigned int>> searchResults = ShapeDescriptor::gpu::findRadialIntersectionCountImagesInHaystack(
                device_haystackImages, device_haystackImages);

        ShapeDescriptor::dump::searchResults<unsigned int>(searchResults, "rici_another_dump.txt");

        SECTION("Equivalent images are the top search results") {
            // First and last image are constant, which causes the pearson correlation to be undefined.
            // We therefore exclude them from the test.
            for (int image = 0; image < imageCount; image++) {
                // Allow for shared first places
                int resultIndex = 0;
                while (searchResults.content[image].scores[resultIndex] == 0) {
                    if (searchResults.content[image].indices[resultIndex] == image) {
                        break;
                    }
                    resultIndex++;
                }
                std::cout << image << std::endl;
                REQUIRE(searchResults.content[image].scores[resultIndex] == 0);
                REQUIRE(searchResults.content[image].indices[resultIndex] == image);
            }
        }

        SECTION("Scores are properly sorted") {
            for(int image = 0; image < imageCount; image++) {
                for (int i = 0; i < SEARCH_RESULT_COUNT - 1; i++) {
                    float firstImageCurrentScore = searchResults.content[image].scores[i];
                    float firstImageNextScore = searchResults.content[image].scores[i + 1];

                    REQUIRE(firstImageCurrentScore <= firstImageNextScore);
                }
            }
        }
    }

    SECTION("Ranking by computing rank indices") {

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                device_haystackImages, device_haystackImages);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        ShapeDescriptor::gpu::array<ShapeDescriptor::RICIDescriptor> device_haystackImages_reversed = ShapeDescriptor::copy::hostArrayToDevice(imageSequence);

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::gpu::computeRadialIntersectionCountImageSearchResultRanks(
                device_haystackImages_reversed, device_haystackImages_reversed);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}