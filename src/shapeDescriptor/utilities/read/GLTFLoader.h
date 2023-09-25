#pragma once

#include <filesystem>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include "RecomputeNormals.h"

namespace ShapeDescriptor {

    namespace utilities {
        cpu::Mesh loadGLTFMesh(std::filesystem::path, ShapeDescriptor::RecomputeNormals recomputeNormals = RecomputeNormals::DO_NOT_RECOMPUTE);
    }
}