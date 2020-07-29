#include "utilities/spinImageGenerator.h"
#include <catch2/catch.hpp>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>
#include <spinImage/gpu/spinImageSearcher.cuh>
#include <iostream>
#include <spinImage/utilities/dumpers/descriptors.h>
#include <spinImage/utilities/copy/array.h>

const float correlationThreshold = 0.00001f;

TEST_CASE("Ranking of Spin Images on the GPU") {

    SpinImage::utilities::createCUDAContext();

    SpinImage::cpu::array<SpinImage::gpu::SpinImageDescriptor> imageSequence = generateKnownSpinImageSequence(imageCount, pixelsPerImage);

    SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_haystackImages = SpinImage::copy::hostArrayToDevice(imageSequence);

    SECTION("Ranking by generating search results on GPU") {
        SpinImage::cpu::array<SpinImage::gpu::SpinImageSearchResults> searchResults = SpinImage::gpu::findSpinImagesInHaystack(device_haystackImages, device_haystackImages);

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

        SpinImage::cpu::array<unsigned int> results = SpinImage::gpu::computeSpinImageSearchResultRanks(device_haystackImages, device_haystackImages);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        SpinImage::gpu::array<SpinImage::gpu::SpinImageDescriptor> device_haystackImages_reversed = SpinImage::copy::hostArrayToDevice(imageSequence);

        SpinImage::cpu::array<unsigned int> results = SpinImage::gpu::computeSpinImageSearchResultRanks(device_haystackImages_reversed, device_haystackImages_reversed);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}