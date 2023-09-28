#pragma once

#include <filesystem>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/PointCloud.h>

namespace ShapeDescriptor {
    namespace utilities {
        void writeCompressedGeometryFile(const ShapeDescriptor::cpu::Mesh &mesh, const std::filesystem::path &filePath, bool stripVertexColours = false);
        void writeCompressedGeometryFile(const ShapeDescriptor::cpu::PointCloud &cloud, const std::filesystem::path &filePath, bool stripVertexColours = false);
    }
}