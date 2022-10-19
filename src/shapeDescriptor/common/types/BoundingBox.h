#pragma once

#include <shapeDescriptor/gpu/types/float3.h>

namespace ShapeDescriptor {
    struct BoundingBox {
        float3 min;
        float3 max;
    };
}

