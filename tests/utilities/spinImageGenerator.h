#pragma once
#include <shapeDescriptor/shapeDescriptor.h>

const int imageCount = spinImageWidthPixels * spinImageWidthPixels + 1 - 2;
const int pixelsPerImage = spinImageWidthPixels * spinImageWidthPixels;

ShapeDescriptor::cpu::array<spinImagePixelType> generateEmptySpinImages(size_t imageCount);
ShapeDescriptor::cpu::array<radialIntersectionCountImagePixelType> generateEmptyRadialIntersectionCountImages(size_t imageCount);

ShapeDescriptor::cpu::array<spinImagePixelType> generateRepeatingTemplateSpinImage(
        spinImagePixelType patternPart0,
        spinImagePixelType patternPart1,
        spinImagePixelType patternPart2,
        spinImagePixelType patternPart3,
        spinImagePixelType patternPart4,
        spinImagePixelType patternPart5,
        spinImagePixelType patternPart6,
        spinImagePixelType patternPart7);
ShapeDescriptor::cpu::array<radialIntersectionCountImagePixelType> generateRepeatingTemplateRadialIntersectionCountImage(
        radialIntersectionCountImagePixelType patternPart0,
        radialIntersectionCountImagePixelType patternPart1,
        radialIntersectionCountImagePixelType patternPart2,
        radialIntersectionCountImagePixelType patternPart3,
        radialIntersectionCountImagePixelType patternPart4,
        radialIntersectionCountImagePixelType patternPart5,
        radialIntersectionCountImagePixelType patternPart6,
        radialIntersectionCountImagePixelType patternPart7);

ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> generateKnownSpinImageSequence(const int imageCount, const int pixelsPerImage);
ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> generateKnownRadialIntersectionCountImageSequence(const int imageCount, const int pixelsPerImage);