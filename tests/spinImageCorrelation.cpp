#include "utilities/spinImageGenerator.h"
#include <catch2/catch.hpp>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/utilities/CUDAContextCreator.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <shapeDescriptor/gpu/spinImageSearcher.cuh>
#include <iostream>
#include <shapeDescriptor/utilities/copy/array.h>

const float correlationThreshold = 0.00001f;

TEST_CASE("Ranking of Spin Images on the GPU") {

    ShapeDescriptor::utilities::createCUDAContext();

    ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SpinImageDescriptor> imageSequence = generateKnownSpinImageSequence(imageCount, pixelsPerImage);

    ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::SpinImageDescriptor> device_haystackImages = ShapeDescriptor::copy::hostArrayToDevice(imageSequence);

    SECTION("Ranking by generating search results on GPU") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::gpu::SpinImageSearchResults> searchResults = ShapeDescriptor::gpu::findSpinImagesInHaystack(device_haystackImages, device_haystackImages);

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

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::gpu::computeSpinImageSearchResultRanks(device_haystackImages, device_haystackImages);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        ShapeDescriptor::gpu::array<ShapeDescriptor::gpu::SpinImageDescriptor> device_haystackImages_reversed = ShapeDescriptor::copy::hostArrayToDevice(imageSequence);

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::gpu::computeSpinImageSearchResultRanks(device_haystackImages_reversed, device_haystackImages_reversed);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}