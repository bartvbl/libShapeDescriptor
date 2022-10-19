#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/gpu/types/array.h>
#include <shapeDescriptor/libraryBuildSettings.h>
#include <stdexcept>

namespace ShapeDescriptor {
    namespace free {
        template<typename T>
        void array(ShapeDescriptor::cpu::array<T> &arrayToFree) {
            delete[] arrayToFree.content;
            arrayToFree.content = nullptr;
        }

        template<typename T>
        void array(ShapeDescriptor::gpu::array<T> &arrayToFree) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            cudaFree(arrayToFree.content);
            arrayToFree.content = nullptr;
#else
            throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
        }
    }
}