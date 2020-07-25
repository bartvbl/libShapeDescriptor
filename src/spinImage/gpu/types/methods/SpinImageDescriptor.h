#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>

namespace SpinImage {
    namespace gpu {
        class SpinImageDescriptor {
            std::array<radialIntersectionCountImagePixelType, spinImageWidthPixels * spinImageWidthPixels> contents;
        };
    }
}

