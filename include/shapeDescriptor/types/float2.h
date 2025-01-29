#pragma once

#include "float3.h"

namespace ShapeDescriptor {
    namespace cpu {

        struct float2 {
            float x;
            float y;

            float2 operator+ (const float2 other) const {
                float2 out;
                out.x = x + other.x;
                out.y = y + other.y;
                return out;
            }
        
            float2 operator- (const float2 other) const {
                float2 out;
                out.x = x - other.x;
                out.y = y - other.y;
                return out;
            }
        
            float2 operator* (float other) const {
                float2 out;
                out.x = other * x;
                out.y = other * y;
                return out;
            }
        
            float2 operator/ (float other) {
                float2 out;
                out.x = x / other;
                out.y = y / other;
                return out;
            }

            float2(float _x, float _y) : x{_x}, y{_y} {}
            float2() = default;
        };
    }
}

inline float length(ShapeDescriptor::cpu::float2 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y);
}

inline ShapeDescriptor::cpu::float2 make_float2_cpu(float x, float y) {
    ShapeDescriptor::cpu::float2 out;
    out.x = x;
    out.y = y;
    return out;
}

inline ShapeDescriptor::cpu::float2 normalize(ShapeDescriptor::cpu::float2 in) {
    ShapeDescriptor::cpu::float2 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    return out;
}

inline float dot(ShapeDescriptor::cpu::float2 a, ShapeDescriptor::cpu::float2 b) {
    return a.x * b.x + a.y * b.y;
}

inline bool operator==(ShapeDescriptor::cpu::float2 a, ShapeDescriptor::cpu::float2 b) {
    return a.x == b.x && a.y == b.y;
}

inline ShapeDescriptor::cpu::float2 operator* (float other, ShapeDescriptor::cpu::float2 in) {
    ShapeDescriptor::cpu::float2 out;
    out.x = other * in.x;
    out.y = other * in.y;
    return out;
}