#pragma once

#include <shapeSearch/common/types/VertexDescriptors.h>
#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>

struct VertexDescriptors {
    array<classicSpinImagePixelType> classicDescriptorArray;
    array<newSpinImagePixelType> newDescriptorArray;

    bool isClassic = false;
    bool isNew = false;
};