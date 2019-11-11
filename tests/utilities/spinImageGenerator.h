#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1 - 2;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

SpinImage::array<spinImagePixelType> generateEmptySpinImages(size_t imageCount);
SpinImage::array<quasiSpinImagePixelType> generateEmptyQuasiSpinImages(size_t imageCount);

SpinImage::array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7);
SpinImage::array<quasiSpinImagePixelType> generateRepeatingTemplateQuasiSpinImage(
        quasiSpinImagePixelType patternPart0,
        quasiSpinImagePixelType patternPart1,
        quasiSpinImagePixelType patternPart2,
        quasiSpinImagePixelType patternPart3,
        quasiSpinImagePixelType patternPart4,
        quasiSpinImagePixelType patternPart5,
        quasiSpinImagePixelType patternPart6,
        quasiSpinImagePixelType patternPart7);

SpinImage::array<spinImagePixelType> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage);
SpinImage::array<quasiSpinImagePixelType> generateKnownQuasiSpinImageSequence(const int imageCount, const int pixelsPerImage);