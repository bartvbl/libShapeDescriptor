#pragma once

#include <shapeSearch/common/types/array.h>

namespace SpinImage {
    namespace cpu {
        void computeMSIRisingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
        void computeMSIFallingHorizontal(array<unsigned int> MSIDescriptor, array<unsigned int> QSIDescriptor);
    }
}