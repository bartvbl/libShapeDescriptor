#pragma once

#include "spinImage/cpu/types/HostMesh.h"

namespace SpinImage {
    namespace utilities {
        HostMesh loadOBJ(std::string src, bool recomputeNormals = false);
    }
}