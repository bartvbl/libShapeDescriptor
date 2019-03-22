#include "correlationComputations.h"
#include <catch2/catch.hpp>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include <spinImage/cpu/spinImageSearcher.h>
#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/utilities/CUDAContextCreator.h>
#include <cuda_runtime.h>
#include <nvidia/helper_cuda.h>

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

TEST_CASE("Correlation computation", "[correlation]") {
    SpinImage::utilities::createCUDAContext();

    SECTION("Equivalent images") {
        array<spinImagePixelType> constantImage =
                generateRepeatingTemplateImage<spinImagePixelType>(0, 1, 0, 1, 0, 1, 0, 1);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(constantImage.content, constantImage.content, 0, 0);

        delete[] constantImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Opposite images") {
        array<spinImagePixelType> positiveImage = generateEmptyImages<spinImagePixelType>(1);
        array<spinImagePixelType> negativeImage = generateEmptyImages<spinImagePixelType>(1);

        for(int i = 0; i < spinImageWidthPixels * spinImageWidthPixels; i++) {
            positiveImage.content[i] = float(i);
            negativeImage.content[i] = float(i) * -1;
        }

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);

        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == -1);
    }



}