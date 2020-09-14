#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadOFF(std::string src, bool recomputeNormals = false);
    }
}