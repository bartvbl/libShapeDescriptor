#pragma once

#include <shapeDescriptor/libraryBuildSettings.h>

namespace ShapeDescriptor {
    struct FPFHDescriptor {
        float contents[3 * FPFH_BINS_PER_FEATURE];
    };
}

