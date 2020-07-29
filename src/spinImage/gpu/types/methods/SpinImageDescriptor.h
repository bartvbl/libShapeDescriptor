#pragma once

#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct SpinImageDescriptor {
            radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

