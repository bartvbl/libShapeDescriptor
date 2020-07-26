#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <array>

namespace SpinImage {
    namespace gpu {
        class SpinImageDescriptor {
        public:
            std::array<radialIntersectionCountImagePixelType, spinImageWidthPixels * spinImageWidthPixels> contents;
        };
    }
}

