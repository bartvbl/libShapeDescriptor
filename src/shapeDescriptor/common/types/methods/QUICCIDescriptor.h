#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    const unsigned int QUICCIDescriptorLength = (spinImageWidthPixels * spinImageWidthPixels) / (sizeof(unsigned int) * 8);

    struct QUICCIDescriptor {
        unsigned int contents[QUICCIDescriptorLength];
    };
}

