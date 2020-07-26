#pragma once

#include <array>

namespace SpinImage {
    namespace gpu {
        struct FPFHDescriptor {
            std::array<float, 3 * FPFH_BINS_PER_FEATURE> contents;
        };
    }
}
