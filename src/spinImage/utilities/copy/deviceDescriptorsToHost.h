#pragma once

#include <spinImage/common/types/array.h>
#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace copy {
        array<quasiSpinImagePixelType>
        QSIDescriptorsToHost(array<quasiSpinImagePixelType> device_descriptors, size_t imageCount);

        array<spinImagePixelType>
        spinImageDescriptorsToHost(array<spinImagePixelType> device_descriptors, size_t imageCount);
    }
}