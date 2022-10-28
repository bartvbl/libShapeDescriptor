#pragma once

#include "shapeDescriptor/gpu/types/float3.h"

namespace ShapeDescriptor {
    struct OrientedPoint {
        // Force using the CUDA float3 type rather than the one defined in libShapeDescriptor
        ::float3 vertex;
        ::float3 normal;
    };
}

