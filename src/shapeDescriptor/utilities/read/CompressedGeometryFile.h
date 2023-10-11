#pragma once

#include <shapeDescriptor/cpu/types/PointCloud.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <filesystem>

namespace ShapeDescriptor {
    ShapeDescriptor::cpu::Mesh readMeshFromCompressedGeometryFile(const std::filesystem::path &filePath);
    ShapeDescriptor::cpu::PointCloud readPointCloudFromCompressedGeometryFile(const std::filesystem::path &filePath);
}