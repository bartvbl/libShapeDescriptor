#pragma once

#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/PointCloud.h>

namespace ShapeDescriptor {
    uint32_t hashMesh(const cpu::Mesh& mesh);
    uint32_t hashPointCloud(const cpu::PointCloud& cloud);
    bool compareMesh(const cpu::Mesh& mesh, const cpu::Mesh& otherMesh);
}