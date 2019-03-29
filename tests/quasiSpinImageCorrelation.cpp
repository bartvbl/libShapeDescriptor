#include "quasiSpinImageCorrelation.h"

#include <catch2/catch.hpp>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/cpu/spinImageSearcher.h>
#include "utilities/spinImageGenerator.h"

TEST_CASE("Basic correlation computation (Quasi Spin Images)", "[correlation]") {

    SECTION("Equivalent images") {
        array<quasiSpinImagePixelType> constantImage =
                generateRepeatingTemplateQuasiSpinImage(0, 1, 0, 1, 0, 1, 0, 1);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(constantImage.content, constantImage.content, 0,
                                                                        0);

        delete[] constantImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Opposite images") {
        array<quasiSpinImagePixelType> positiveImage = generateEmptyQuasiSpinImages(1);
        array<quasiSpinImagePixelType> negativeImage = generateEmptyQuasiSpinImages(1);

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

    SECTION("Equivalent constant images") {
        array<quasiSpinImagePixelType> positiveImage = generateRepeatingTemplateQuasiSpinImage(
                5, 5, 5, 5, 5, 5, 5, 5);
        array<quasiSpinImagePixelType> negativeImage = generateRepeatingTemplateQuasiSpinImage(
                5, 5, 5, 5, 5, 5, 5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0,
                                                                        0);
        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 1);
    }

    SECTION("Different constant images") {
        array<quasiSpinImagePixelType> positiveImage = generateRepeatingTemplateQuasiSpinImage(
                2, 2, 2, 2, 2, 2, 2, 2);
        array<quasiSpinImagePixelType> negativeImage = generateRepeatingTemplateQuasiSpinImage(
                5, 5, 5, 5, 5, 5, 5, 5);

        float correlation = SpinImage::cpu::computeImagePairCorrelation(positiveImage.content, negativeImage.content, 0, 0);

        float otherCorrelation = SpinImage::cpu::computeImagePairCorrelation(negativeImage.content, positiveImage.content, 0, 0);

        delete[] positiveImage.content;
        delete[] negativeImage.content;
        REQUIRE(correlation == 0.4f);
        REQUIRE(otherCorrelation == 0.4f);
    }
}