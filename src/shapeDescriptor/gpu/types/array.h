#pragma once

#include <cstddef>

namespace SpinImage {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;
        };
    }
}