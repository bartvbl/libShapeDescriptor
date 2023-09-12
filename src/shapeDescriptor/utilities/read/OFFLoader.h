#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <filesystem>

namespace ShapeDescriptor {
    namespace utilities {
        // has no option to recompute normals because the file format does not support them
        cpu::Mesh loadOFF(std::filesystem::path src);
    }
}