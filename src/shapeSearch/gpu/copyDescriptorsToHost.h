#pragma once

#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>

array<newSpinImagePixelType> copyQSIDescriptorsToHost(array<newSpinImagePixelType> device_descriptors, size_t imageCount);