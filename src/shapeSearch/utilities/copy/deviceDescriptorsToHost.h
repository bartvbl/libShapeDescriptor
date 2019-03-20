#pragma once

#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>

namespace SpinImage {
    namespace copy {
        array<newSpinImagePixelType>
        QSIDescriptorsToHost(array<newSpinImagePixelType> device_descriptors, size_t imageCount);

        array<classicSpinImagePixelType>
        spinImageDescriptorsToHost(array<classicSpinImagePixelType> device_descriptors, size_t imageCount);
    }
}