#pragma once

#include <spinImage/libraryBuildSettings.h>

namespace SpinImage {
    namespace gpu {
        struct FPFHDescriptor {
            float contents[3 * FPFH_BINS_PER_FEATURE];
        };
    }
}

