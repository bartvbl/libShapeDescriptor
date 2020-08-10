#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct SpinImageDescriptor {
            spinImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

