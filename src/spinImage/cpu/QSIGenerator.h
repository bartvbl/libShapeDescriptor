#pragma once

#include <algorithm>
#include <spinImage/cpu/types/CPURasterisationSettings.h>
#include <spinImage/cpu/types/float3_cpu.h>
#include <spinImage/common/buildSettings/derivedBuildSettings.h>
#include "spinImage/utilities/OBJLoader.h"
#include "spinImage/common/types/array.h"

namespace SpinImage {
    namespace cpu {
        void generateQuasiSpinImage(
                array<quasiSpinImagePixelType> descriptor,
                CPURasterisationSettings settings);

        array<quasiSpinImagePixelType> generateQuasiSpinImages(
                CPURasterisationSettings settings);
    }
}