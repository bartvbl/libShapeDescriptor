#pragma once

#include <cstddef>
#include <array>
#include <spinImage/libraryBuildSettings.h>
#include "QuiccImage.h"

namespace SpinImage {
    namespace cpu {
        struct QUICCIImages {
            QuiccImage* horizontallyIncreasingImages;
            QuiccImage* horizontallyDecreasingImages;
            size_t imageCount;
        };
    }
}