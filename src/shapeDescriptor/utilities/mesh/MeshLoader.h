#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"
#include <experimental/filesystem>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadMesh(std::experimental::filesystem::path src, bool recomputeNormals = false);
    }
}