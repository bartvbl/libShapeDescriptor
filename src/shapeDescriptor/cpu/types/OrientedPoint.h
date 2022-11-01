#pragma once

#include "float3.h"

namespace ShapeDescriptor {
    namespace cpu {
        struct OrientedPoint {
            ShapeDescriptor::cpu::float3 vertex;
            ShapeDescriptor::cpu::float3 normal;
        };
    }
}