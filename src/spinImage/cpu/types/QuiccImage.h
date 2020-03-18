#pragma once

#include <array>
#include <spinImage/libraryBuildSettings.h>

typedef std::array<unsigned int, UINTS_PER_QUICCI> QuiccImage;

static QuiccImage combineQuiccImages(
        const QuiccImage &image1,
        const QuiccImage &image2) {
    QuiccImage combinedImage;
    for(unsigned int i = 0; i < 128; i++) {
        combinedImage[i] = image1[i] | image2[i];
    }
    return combinedImage;
}