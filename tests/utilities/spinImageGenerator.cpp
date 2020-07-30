#include <shapeDescriptor/libraryBuildSettings.h>
#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/methods/RICIDescriptor.h>
#include <shapeDescriptor/gpu/types/methods/SpinImageDescriptor.h>
#include "spinImageGenerator.h"


template<typename pixelType>
SpinImage::cpu::array<pixelType> generateEmptyImages(size_t imageCount) {
    pixelType* image = new pixelType[imageCount * spinImageWidthPixels * spinImageWidthPixels];
    SpinImage::cpu::array<pixelType> images;
    images.content = image;
    images.length = imageCount * spinImageWidthPixels * spinImageWidthPixels;

    return images;
}

template<typename pixelType>
SpinImage::cpu::array<pixelType> generateRepeatingTemplateImage(
        pixelType patternPart0,
        pixelType patternPart1,
        pixelType patternPart2,
        pixelType patternPart3,
        pixelType patternPart4,
        pixelType patternPart5,
        pixelType patternPart6,
        pixelType patternPart7) {

    SpinImage::cpu::array<pixelType> images = generateEmptyImages<pixelType>(1);

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
SpinImage::cpu::array<pixelType> generateKnownImageSequence(const int imageCount, const int pixelsPerImage) {
    SpinImage::cpu::array<pixelType> imageSequence = generateEmptyImages<pixelType>(imageCount);

    for(int image = 0; image < imageCount; image++) {
        for(int highIndex = 0; highIndex <= image; highIndex++) {
            imageSequence.content[image].contents[highIndex] = 1;
        }
        for(int lowIndex = image + 1; lowIndex < pixelsPerImage; lowIndex++) {
            imageSequence.content[image].contents[lowIndex] = 0;
        }
    }

    imageSequence.length = imageCount;
    return imageSequence;
}

SpinImage::cpu::array<spinImagePixelType> generateEmptySpinImages(size_t imageCount) {
    return generateEmptyImages<spinImagePixelType>(imageCount);
}

SpinImage::cpu::array<radialIntersectionCountImagePixelType> generateEmptyRadialIntersectionCountImages(size_t imageCount) {
    return generateEmptyImages<radialIntersectionCountImagePixelType>(imageCount);
}

SpinImage::cpu::array<spinImagePixelType> generateRepeatingTemplateSpinImage(
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

SpinImage::cpu::array<radialIntersectionCountImagePixelType> generateRepeatingTemplateRadialIntersectionCountImage(
        radialIntersectionCountImagePixelType patternPart0,
        radialIntersectionCountImagePixelType patternPart1,
        radialIntersectionCountImagePixelType patternPart2,
        radialIntersectionCountImagePixelType patternPart3,
        radialIntersectionCountImagePixelType patternPart4,
        radialIntersectionCountImagePixelType patternPart5,
        radialIntersectionCountImagePixelType patternPart6,
        radialIntersectionCountImagePixelType patternPart7) {
    return generateRepeatingTemplateImage<radialIntersectionCountImagePixelType>(
            patternPart0,
            patternPart1,
            patternPart2,
            patternPart3,
            patternPart4,
            patternPart5,
            patternPart6,
            patternPart7);
}

SpinImage::cpu::array<SpinImage::gpu::SpinImageDescriptor> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage) {
    return generateKnownImageSequence<SpinImage::gpu::SpinImageDescriptor>(imageCount, pixelsPerImage);
}

SpinImage::cpu::array<SpinImage::gpu::RICIDescriptor> generateKnownRadialIntersectionCountImageSequence(const int imageCount, const int pixelsPerImage) {
    return generateKnownImageSequence<SpinImage::gpu::RICIDescriptor>(imageCount, pixelsPerImage);
}


