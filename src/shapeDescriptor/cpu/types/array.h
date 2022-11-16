#pragma once

#include <cstddef>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/utilities/copy/array.h>

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

            array(size_t length, TYPE* content)
                : length(length),
                  content(content) {}

            ShapeDescriptor::gpu::array<TYPE> toGPU() {
                return ShapeDescriptor::copy::hostArrayToDevice(this);
            }
        };


    }
}