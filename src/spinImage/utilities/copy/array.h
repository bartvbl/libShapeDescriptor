#pragma once

#include <spinImage/cpu/types/array.h>
#include <spinImage/gpu/types/array.h>

namespace SpinImage {
    namespace copy {
        template<typename T>
        SpinImage::cpu::array<T> deviceArrayToHost(SpinImage::gpu::array<T> array);

        template<typename T>
        SpinImage::gpu::array<T> hostArrayToDevice(SpinImage::cpu::array<T> array);
    }
}