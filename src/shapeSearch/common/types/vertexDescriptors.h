#pragma once

#include <shapeSearch/common/types/vertexDescriptors.h>
#include <shapeSearch/common/types/arrayTypes.hpp>
#include <shapeSearch/libraryBuildSettings.h>

struct VertexDescriptors {
    array<classicSpinImagePixelType> classicDescriptorArray;
    array<newSpinImagePixelType> newDescriptorArray;
    array<unsigned int> microshapeArray;
    // unsigned byte variant is for compressed versions.
    //unsigned char* descriptorArray;
    bool isClassic = false;
    bool isNew = false;
    bool isMicroshape = false;
};