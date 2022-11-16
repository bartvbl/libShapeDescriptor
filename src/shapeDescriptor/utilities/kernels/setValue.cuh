#pragma once

namespace ShapeDescriptor {
    namespace internal {
        void gpuMemsetMultibyte(char* array, size_t length, const char* value, size_t valueSize);
    }

    namespace gpu {
        template<typename TYPE>
        void setValue(TYPE* array, size_t length, TYPE value) {
#ifdef DESCRIPTOR_CUDA_KERNELS_ENABLED
            // A function boundary is necessary to ensure the associated GPU kernel can be called when this template is
            // included from a regular non-CUDA source file
            internal::gpuMemsetMultibyte(reinterpret_cast<char*>(array), length, reinterpret_cast<char*>(&value), sizeof(value));
#else
            throw std::runtime_error(ShapeDescriptor::cudaMissingErrorMessage);
#endif
        }
    }
}