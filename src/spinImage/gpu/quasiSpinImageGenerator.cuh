#pragma once

#include <spinImage/libraryBuildSettings.h>
#include <spinImage/gpu/types/DeviceMesh.h>


namespace SpinImage {
    namespace gpu {
        array<quasiSpinImagePixelType> generateQuasiSpinImages(
                DeviceMesh device_mesh,
                float spinImageWidth,
                SpinImage::debug::QSIRunInfo* runinfo = nullptr);
    }
}