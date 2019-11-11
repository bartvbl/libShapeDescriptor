#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/common/types/array.h>

namespace SpinImage {
    namespace copy {
        array<radialIntersectionCountImagePixelType> hostDescriptorsToDevice(array<radialIntersectionCountImagePixelType> hostDescriptors, size_t imageCount);
        array<spinImagePixelType> hostDescriptorsToDevice(array<spinImagePixelType> hostDescriptors, size_t imageCount);
    }
}