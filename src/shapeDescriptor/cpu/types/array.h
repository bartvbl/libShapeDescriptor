#pragma once

#include <cstddef>
#include <cassert>
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

            ShapeDescriptor::gpu::array<TYPE> copyToGPU() {
                return ShapeDescriptor::copy::hostArrayToDevice<TYPE>({length, content});
            }

            void setValue(TYPE &value) {
                std::fill_n(content, length, value);
            }

            TYPE& operator[](size_t index) {
                assert(index); // >= 0
                assert(index < length);
                return *(content + index);
            }
        };


    }
}