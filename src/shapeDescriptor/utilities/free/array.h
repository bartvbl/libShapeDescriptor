#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>

namespace ShapeDescriptor {
    namespace free {
        template<typename T>
        void array(ShapeDescriptor::cpu::array<T> &arrayToFree) {
            delete[] arrayToFree.content;
            arrayToFree.content = nullptr;
        }

        template<typename T>
        void array(ShapeDescriptor::gpu::array<T> &arrayToFree) {
            cudaFree(arrayToFree.content);
            arrayToFree.content = nullptr;
        }
    }
}