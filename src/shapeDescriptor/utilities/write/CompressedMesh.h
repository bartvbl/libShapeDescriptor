#pragma once

#include <filesystem>
#include <shapeDescriptor/cpu/types/Mesh.h>

namespace ShapeDescriptor {
    namespace utilities {
        void writeCompressedMesh(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours = false);
    }
}