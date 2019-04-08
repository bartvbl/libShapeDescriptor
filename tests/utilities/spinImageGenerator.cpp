#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>
#include "spinImageGenerator.h"


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

template<typename pixelType>
array<pixelType> generateKnownImageSequence(const int imageCount, const int pixelsPerImage) {
    array<pixelType> imageSequence = generateEmptyImages<pixelType>(imageCount);

    for(int image = 0; image < imageCount; image++) {
        for(int highIndex = 0; highIndex <= image; highIndex++) {
            imageSequence.content[image * pixelsPerImage + highIndex] = 1;
        }
        for(int lowIndex = image + 1; lowIndex < pixelsPerImage; lowIndex++) {
            imageSequence.content[image * pixelsPerImage + lowIndex] = 0;
        }
    }

    imageSequence.length = imageCount;
    return imageSequence;
}

array<spinImagePixelType> generateEmptySpinImages(size_t imageCount) {
    return generateEmptyImages<spinImagePixelType>(imageCount);
}

array<quasiSpinImagePixelType> generateEmptyQuasiSpinImages(size_t imageCount) {
    return generateEmptyImages<quasiSpinImagePixelType>(imageCount);
}

array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7) {
    return generateRepeatingTemplateImage<spinImagePixelType>(
            patternPart0,
            patternPart1,
            patternPart2,
            patternPart3,
            patternPart4,
            patternPart5,
            patternPart6,
            patternPart7);
}

array<quasiSpinImagePixelType> generateRepeatingTemplateQuasiSpinImage(
        quasiSpinImagePixelType patternPart0,
        quasiSpinImagePixelType patternPart1,
        quasiSpinImagePixelType patternPart2,
        quasiSpinImagePixelType patternPart3,
        quasiSpinImagePixelType patternPart4,
        quasiSpinImagePixelType patternPart5,
        quasiSpinImagePixelType patternPart6,
        quasiSpinImagePixelType patternPart7) {
    return generateRepeatingTemplateImage<quasiSpinImagePixelType>(
            patternPart0,
            patternPart1,
            patternPart2,
            patternPart3,
            patternPart4,
            patternPart5,
            patternPart6,
            patternPart7);
}

array<spinImagePixelType> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage) {
    return generateKnownImageSequence<spinImagePixelType>(imageCount, pixelsPerImage);
}

array<quasiSpinImagePixelType> generateKnownQuasiSpinImageSequence(const int imageCount, const int pixelsPerImage) {
    return generateKnownImageSequence<quasiSpinImagePixelType>(imageCount, pixelsPerImage);
}


