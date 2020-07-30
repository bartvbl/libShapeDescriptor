#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>

namespace SpinImage {
    namespace utilities {
        SpinImage::cpu::Mesh fitMeshInsideSphereOfRadius(SpinImage::cpu::Mesh &input, float radius);
    }
}
