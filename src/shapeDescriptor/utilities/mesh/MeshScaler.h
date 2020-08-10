#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh scaleMesh(cpu::Mesh &model, cpu::Mesh &scaledModel, float spinImagePixelSize);

        ShapeDescriptor::cpu::Mesh fitMeshInsideSphereOfRadius(ShapeDescriptor::cpu::Mesh &input, float radius);
    }
}