#pragma once

#include <cstddef>

namespace ShapeDescriptor {
    namespace cpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;
        };
    }
}