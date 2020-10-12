#pragma once

#include <cstddef>

namespace ShapeDescriptor {
    namespace cpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;

            array() {}

            array(size_t length)
                : length(length),
                  content(new TYPE[length]) {}
        };
    }
}