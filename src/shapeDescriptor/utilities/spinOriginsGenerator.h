#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/common/types/OrientedPoint.h>

namespace ShapeDescriptor {
    namespace utilities {
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> generateSpinOriginBuffer(cpu::Mesh &mesh);
        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> generateUniqueSpinOriginBuffer(cpu::Mesh &mesh);
    }
}