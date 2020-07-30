#pragma once
#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/array.h>
#include <spinImage/cpu/types/array.h>
#include <spinImage/gpu/types/methods/SpinImageDescriptor.h>
#include <spinImage/gpu/types/methods/RICIDescriptor.h>

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1 - 2;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

SpinImage::cpu::array<spinImagePixelType> generateEmptySpinImages(size_t imageCount);
SpinImage::cpu::array<radialIntersectionCountImagePixelType> generateEmptyRadialIntersectionCountImages(size_t imageCount);

SpinImage::cpu::array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7);
SpinImage::cpu::array<radialIntersectionCountImagePixelType> generateRepeatingTemplateRadialIntersectionCountImage(
        radialIntersectionCountImagePixelType patternPart0,
        radialIntersectionCountImagePixelType patternPart1,
        radialIntersectionCountImagePixelType patternPart2,
        radialIntersectionCountImagePixelType patternPart3,
        radialIntersectionCountImagePixelType patternPart4,
        radialIntersectionCountImagePixelType patternPart5,
        radialIntersectionCountImagePixelType patternPart6,
        radialIntersectionCountImagePixelType patternPart7);

SpinImage::cpu::array<SpinImage::gpu::SpinImageDescriptor> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage);
SpinImage::cpu::array<SpinImage::gpu::RICIDescriptor> generateKnownRadialIntersectionCountImageSequence(const int imageCount, const int pixelsPerImage);