#pragma once

#include <shapeDescriptor/cpu/types/array.h>
#include <shapeDescriptor/cpu/types/Mesh.h>
#include <shapeDescriptor/cpu/types/OrientedPoint.h>

namespace ShapeDescriptor {
    namespace cpu {
        ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint> generateSpinOriginBuffer(cpu::Mesh &mesh);
        ShapeDescriptor::cpu::array<ShapeDescriptor::cpu::OrientedPoint> generateUniqueSpinOriginBuffer(cpu::Mesh &mesh);
    }
}