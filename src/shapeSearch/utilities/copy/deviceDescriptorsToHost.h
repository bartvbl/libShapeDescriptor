#pragma once

#include <shapeSearch/common/types/array.h>
#include <shapeSearch/libraryBuildSettings.h>

namespace SpinImage {
    namespace copy {
        array<quasiSpinImagePixelType>
        QSIDescriptorsToHost(array<quasiSpinImagePixelType> device_descriptors, size_t imageCount);

        array<spinImagePixelType>
        spinImageDescriptorsToHost(array<spinImagePixelType> device_descriptors, size_t imageCount);
    }
}