#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadOBJ(std::string src, bool recomputeNormals = false);
    }
}