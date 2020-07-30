#pragma once

#include "spinImage/cpu/types/Mesh.h"

namespace SpinImage {
    namespace utilities {
        cpu::Mesh loadOBJ(std::string src, bool recomputeNormals = false);
    }
}