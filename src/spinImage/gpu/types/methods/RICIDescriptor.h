#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>

namespace SpinImage {
    namespace gpu {
        class RICIDescriptor {
            std::array<radialIntersectionCountImagePixelType, spinImageWidthPixels * spinImageWidthPixels> contents;
        };
    }
}

