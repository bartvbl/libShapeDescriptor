#pragma once

#include <cstddef>

namespace SpinImage {
    namespace gpu {
        struct QUICCIImages {
            unsigned int* images;
            size_t imageCount;
        };
    }
}