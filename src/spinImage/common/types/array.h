#pragma once

#include <cstddef>

namespace SpinImage {

    template<typename TYPE> struct array
    {
        size_t length;
        TYPE* content;
    };
}