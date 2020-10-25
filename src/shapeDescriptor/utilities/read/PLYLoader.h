#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"

// Note: this is not a complete implementation of the file format.
// Instead, it should capture the parts that are used most in practice.

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadPLY(std::string src, bool recomputeNormals = false);
    }
}

