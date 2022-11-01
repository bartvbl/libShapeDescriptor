#pragma once

#include <shapeDescriptor/cpu/types/float3.h>
#include <iostream>

namespace ShapeDescriptor {
    struct OrientedPoint {
        ShapeDescriptor::cpu::float3 vertex;
        ShapeDescriptor::cpu::float3 normal;

        bool operator==(ShapeDescriptor::OrientedPoint other) const {
            return vertex == other.vertex && normal == other.normal;
        }
    };

    inline std::ostream & operator<<(std::ostream &os, const ShapeDescriptor::OrientedPoint point) {
        os << "OrientedPoint (vertex: " << &point.vertex <<  ", normal: " << &point.normal << ")";
        return os;
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::OrientedPoint>
    {
        size_t operator()(const ShapeDescriptor::OrientedPoint& p) const
        {
            return std::hash<ShapeDescriptor::cpu::float3>()(p.vertex) ^ std::hash<ShapeDescriptor::cpu::float3>()(p.normal);
        }
    };
}

