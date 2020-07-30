#pragma once

#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct SpinImageDescriptor {
            spinImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

