#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>

namespace SpinImage {
    namespace utilities {
        cpu::Mesh scaleMesh(cpu::Mesh &model, cpu::Mesh &scaledModel, float spinImagePixelSize);

        SpinImage::cpu::Mesh fitMeshInsideSphereOfRadius(SpinImage::cpu::Mesh &input, float radius);
    }
}