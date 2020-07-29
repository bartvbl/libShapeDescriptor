#pragma once

#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct RICIDescriptor {
            radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

