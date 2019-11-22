#pragma once

#include <cstddef>

namespace SpinImage {
    namespace cpu {
        struct QUICCIImages {
            unsigned int* horizontallyIncreasingImages;
            unsigned int* horizontallyDecreasingImages;
            size_t imageCount;
        };
    }
}