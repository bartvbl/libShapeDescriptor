#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct RICIDescriptor {
            radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

