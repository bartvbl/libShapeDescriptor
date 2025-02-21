#pragma once

#include <ostream>
#include <string>
#include <cmath>


namespace ShapeDescriptor {
    namespace cpu {
        struct float4 {
            float x = 0;
            float y = 0;
            float z = 0;
            float w = 0;

            explicit float4() = default;
            explicit float4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

            float4 operator- (float4 other) {
                float4 out;
                out.x = x - other.x;
                out.y = y - other.y;
                out.z = z - other.z;
                out.w = w - other.w;
                return out;
            }

            float4 operator+ (float4 other) {
                float4 out;
                out.x = x + other.x;
                out.y = y + other.y;
                out.z = z + other.z;
                out.w = w + other.w;
                return out;
            }

            float4 operator* (float4 other) {
                float4 out;
                out.x = other.x * x;
                out.y = other.y * y;
                out.z = other.z * z;
                out.w = other.w * w;
                return out;
            }

            float4 operator/ (float divisor) {
                float4 out;
                out.x = x / divisor;
                out.y = y / divisor;
                out.z = z / divisor;
                out.w = w / divisor;
                return out;
            }

            bool operator== (float4 other) {
                return
                        (x == other.x) &&
                        (y == other.y) &&
                        (z == other.z) &&
                        (w == other.w);
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ", " + std::to_string(w) + ")";
            }
        };
    }
}


inline ShapeDescriptor::cpu::float4 make_float4_cpu(float x, float y, float z, float w) {
    ShapeDescriptor::cpu::float4 out;
    out.x = x;
    out.y = y;
    out.z = z;
    out.w = w;
    return out;
}

inline float length(ShapeDescriptor::cpu::float4 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
}

inline ShapeDescriptor::cpu::float4 normalize(ShapeDescriptor::cpu::float4 in) {
    ShapeDescriptor::cpu::float4 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    out.w = in.w / len;
    return out;
}

inline ShapeDescriptor::cpu::float4 operator* (ShapeDescriptor::cpu::float4 vec, float other) {
    ShapeDescriptor::cpu::float4 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    out.w = vec.w * other;
    return out;
}

inline ShapeDescriptor::cpu::float4 operator*(float other, ShapeDescriptor::cpu::float4 vec) {
    return operator*(vec, other);
}

inline std::ostream & operator<<(std::ostream & os, ShapeDescriptor::cpu::float4 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << ")";
    return os;
}
