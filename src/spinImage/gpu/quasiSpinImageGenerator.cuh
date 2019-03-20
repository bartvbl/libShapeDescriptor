#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/DeviceMesh.h>

#include "cuda_runtime.h"

namespace SpinImage {
    namespace gpu {
        array<quasiSpinImagePixelType> generateQuasiSpinImages(
                DeviceMesh device_mesh,
                cudaDeviceProp device_information,
                float spinImageWidth);
    }
}