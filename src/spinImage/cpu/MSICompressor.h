#pragma once

#include "spinImage/common/types/array.h"

namespace SpinImage {
    namespace cpu {
        void convertQSIToMSIImage(array<unsigned int> msiImage, unsigned long long *compressedImage);
    }
}