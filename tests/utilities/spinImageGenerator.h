#pragma once
#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1 - 2;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

array<spinImagePixelType> generateEmptySpinImages(size_t imageCount);
array<quasiSpinImagePixelType> generateEmptyQuasiSpinImages(size_t imageCount);

array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7);
array<quasiSpinImagePixelType> generateRepeatingTemplateQuasiSpinImage(
        quasiSpinImagePixelType patternPart0,
        quasiSpinImagePixelType patternPart1,
        quasiSpinImagePixelType patternPart2,
        quasiSpinImagePixelType patternPart3,
        quasiSpinImagePixelType patternPart4,
        quasiSpinImagePixelType patternPart5,
        quasiSpinImagePixelType patternPart6,
        quasiSpinImagePixelType patternPart7);

array<spinImagePixelType> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage);
array<quasiSpinImagePixelType> generateKnownQuasiSpinImageSequence(const int imageCount, const int pixelsPerImage);