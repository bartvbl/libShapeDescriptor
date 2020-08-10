#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    namespace gpu {
        struct QUICCIDescriptor {
            unsigned int contents[(spinImageWidthPixels * spinImageWidthPixels) / (sizeof(unsigned int) * 8)];
        };
    }
}

