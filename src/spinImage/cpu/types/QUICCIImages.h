#pragma once

#include <cstddef>
#include <array>
#include <spinImage/libraryBuildSettings.h>
#include "QuiccImage.h"

namespace SpinImage {
    namespace cpu {
        struct QUICCIImages {
            QuiccImage* images;
            size_t imageCount;
        };
    }
}