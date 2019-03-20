#pragma once

#include "shapeSearch/common/types/array.h"

namespace SpinImage {
    namespace cpu {
        void convertQSIToMSIImage(array<unsigned int> msiImage, unsigned long long *compressedImage);
    }
}