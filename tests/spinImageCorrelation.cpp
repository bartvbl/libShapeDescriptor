#include <catch2/catch_test_macros.hpp>
#include <shapeDescriptor/shapeDescriptor.h>
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
#include "utilities/spinImageGenerator.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iostream>
#include <algorithm>

const float correlationThreshold = 0.00001f;

TEST_CASE("Ranking of Spin Images on the GPU") {

    ShapeDescriptor::createCUDAContext();

    ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> imageSequence = generateKnownSpinImageSequence(imageCount, pixelsPerImage);

    ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackImages = ShapeDescriptor::copyToGPU(imageSequence);

    SECTION("Ranking by generating search results on GPU") {
        ShapeDescriptor::cpu::array<ShapeDescriptor::SearchResults<float>> searchResults = ShapeDescriptor::findSpinImagesInHaystack(device_haystackImages, device_haystackImages);

        SECTION("Equivalent images are the top search results") {
            for (int image = 0; image < imageCount; image++) {
                // Allow for shared first places
                int resultIndex = 0;
                while (std::abs(searchResults.content[image].scores[resultIndex] - 1.0f) < correlationThreshold) {
                    if (searchResults.content[image].indices[resultIndex] == image) {
                        break;
                    }
                    resultIndex++;
                }

                REQUIRE(std::abs(searchResults.content[image].scores[resultIndex] - 1.0f) < correlationThreshold);
                REQUIRE(searchResults.content[image].indices[resultIndex] == image);
            }
        }

        SECTION("Scores are properly sorted") {
            for(int image = 0; image < imageCount; image++) {
                for (int i = 0; i < SEARCH_RESULT_COUNT - 1; i++) {
                    float firstImageCurrentScore = searchResults.content[image].scores[i];
                    float firstImageNextScore = searchResults.content[image].scores[i + 1];

                    REQUIRE(firstImageCurrentScore >= firstImageNextScore);
                }
            }
        }
    }

    SECTION("Ranking by computing rank indices") {

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::computeSpinImageSearchResultRanks(device_haystackImages, device_haystackImages);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

    }

    SECTION("Ranking by computing rank indices, reversed image sequence") {
        std::reverse(imageSequence.content, imageSequence.content + imageCount * pixelsPerImage);

        ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> device_haystackImages_reversed = ShapeDescriptor::copyToGPU(imageSequence);

        ShapeDescriptor::cpu::array<unsigned int> results = ShapeDescriptor::computeSpinImageSearchResultRanks(device_haystackImages_reversed, device_haystackImages_reversed);

        for(int i = 0; i < imageCount; i++) {
            REQUIRE(results.content[i] == 0);
        }

        cudaFree(device_haystackImages_reversed.content);
    }

    cudaFree(device_haystackImages.content);

}
#endif
