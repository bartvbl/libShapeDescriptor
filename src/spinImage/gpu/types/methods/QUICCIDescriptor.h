#pragma once

#include <spinImage/cpu/types/QuiccImage.h>

namespace SpinImage {
    namespace gpu {
        struct QUICCIDescriptor {
            unsigned int contents[(spinImageWidthPixels * spinImageWidthPixels) / (sizeof(unsigned int) * 8)];
        };
    }
}

