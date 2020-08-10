#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    struct SpinImageDescriptor {
        spinImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
    };
}

