#pragma once

#include <shapeSearch/libraryBuildSettings.h>
#include <shapeSearch/gpu/types/DeviceMesh.h>

namespace SpinImage {
    namespace gpu {
        array<classicSpinImagePixelType>
        generateSpinImages(DeviceMesh device_mesh, cudaDeviceProp device_information, float spinImageWidth,
                           size_t sampleCount);
    }
}