#pragma once

#include <cstddef>

namespace ShapeDescriptor {
    namespace gpu {
        template<typename TYPE> struct array
        {
            size_t length;
            TYPE* content;
        };
    }
}