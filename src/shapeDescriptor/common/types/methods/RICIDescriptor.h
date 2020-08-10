#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct RICIDescriptor {
            radialIntersectionCountImagePixelType contents[spinImageWidthPixels * spinImageWidthPixels];
        };
    }
}

