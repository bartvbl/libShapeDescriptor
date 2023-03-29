#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>
#include <vector>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> generateSpinOriginBuffer(const cpu::Mesh &mesh);
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> generateUniqueSpinOriginBuffer(const cpu::Mesh &mesh, std::vector<size_t>* indexMapping = nullptr);
    }
}