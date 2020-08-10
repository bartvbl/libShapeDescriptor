#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    struct RICIDescriptor {
        radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
    };
}

