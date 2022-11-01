#pragma once

#include "float3.h"

namespace ShapeDescriptor {
    namespace cpu {
        struct OrientedPoint {
            ShapeDescriptor::cpu::float3 vertex;
            ShapeDescriptor::cpu::float3 normal;

            bool operator==(ShapeDescriptor::cpu::OrientedPoint other) const {
                return vertex == other.vertex && normal == other.normal;
            }
        };
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::cpu::OrientedPoint>
    {
        size_t operator()(const ShapeDescriptor::cpu::OrientedPoint& p) const
        {
            return std::hash<ShapeDescriptor::cpu::float3>()(p.vertex) ^ std::hash<ShapeDescriptor::cpu::float3>()(p.normal);
        }
    };
}

inline std::ostream & operator<<(std::ostream & os, const ShapeDescriptor::cpu::OrientedPoint point) {
    os << "OrientedPoint (vertex: " << point.vertex <<  ", normal: " << point.normal << ")";
    return os;
}