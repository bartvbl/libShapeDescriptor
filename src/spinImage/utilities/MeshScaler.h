#pragma once

#include <spinImage/cpu/types/HostMesh.h>

namespace SpinImage {
    namespace utilities {
        HostMesh scaleHostMesh(HostMesh &model, HostMesh &scaledModel, float spinImagePixelSize);
    }
}