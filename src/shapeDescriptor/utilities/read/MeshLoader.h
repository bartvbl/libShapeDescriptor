#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"
#include <filesystem>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadMesh(std::filesystem::path src, bool recomputeNormals = false);
    }
}