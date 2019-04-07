#include "quasiSpinImageCorrelation.h"

#include <catch2/catch.hpp>
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>
#include <spinImage/cpu/spinImageSearcher.h>
#include <spinImage/cpu/quasiSpinImageSearcher.h>
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