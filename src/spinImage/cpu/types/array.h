#pragma once

#include <cstddef>

namespace SpinImage {
    namespace cpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;
        };
    }
}