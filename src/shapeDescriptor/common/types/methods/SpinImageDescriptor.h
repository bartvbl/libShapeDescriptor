#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct SpinImageDescriptor {
            spinImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

