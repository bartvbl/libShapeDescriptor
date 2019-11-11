#pragma once

#include <spinImage/cpu/types/Mesh.h>

namespace SpinImage {
    namespace utilities {
        cpu::Mesh scaleMesh(cpu::Mesh &model, cpu::Mesh &scaledModel, float spinImagePixelSize);
    }
}