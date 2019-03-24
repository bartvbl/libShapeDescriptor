#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/DeviceMesh.h>

namespace SpinImage {
    namespace gpu {
        array<spinImagePixelType> generateSpinImages(
                DeviceMesh device_mesh,
                cudaDeviceProp device_information,
                float spinImageWidth,
                size_t sampleCount);
    }
}