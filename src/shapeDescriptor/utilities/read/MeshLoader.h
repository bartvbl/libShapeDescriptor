#pragma once

#include "shapeDescriptor/cpu/types/Mesh.h"
#include "RecomputeNormals.h"
#include <filesystem>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadMesh(std::filesystem::path src, RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);
    }
}