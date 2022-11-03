#pragma once
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <filesystem>

namespace ShapeDescriptor {
    namespace utilities {
        cpu::Mesh loadOFF(std::filesystem::path src);
    }
}