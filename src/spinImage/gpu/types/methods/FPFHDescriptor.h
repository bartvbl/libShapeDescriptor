#pragma once

#include <array>

namespace SpinImage {
    namespace gpu {
        struct FPFHDescriptor {
            std::array<float, 33> contents;
        };
    }
}

