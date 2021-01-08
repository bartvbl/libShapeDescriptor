#pragma once

#include <ostream>
#include <string>
#include <cmath>


namespace ShapeDescriptor {
    namespace cpu {
        struct float3 {
            float x = 0;
            float y = 0;
            float z = 0;

            float3() = default;
            float3(float x, float y, float z) : x(x), y(y), z(z) {}

            float3 operator- (float3 other) {
                float3 out;
                out.x = x - other.x;
                out.y = y - other.y;
                out.z = z - other.z;
                return out;
            }

            float3 operator+ (float3 other) {
                float3 out;
                out.x = x + other.x;
                out.y = y + other.y;
                out.z = z + other.z;
                return out;
            }

            float3 operator* (float3 other) {
                float3 out;
                out.x = other.x * x;
                out.y = other.y * y;
                out.z = other.z * z;
                return out;
            }

            float3 operator/ (float divisor) {
                float3 out;
                out.x = x / divisor;
                out.y = y / divisor;
                out.z = z / divisor;
                return out;
            }

            bool operator== (const float3 &other) const {
                return
                        (x == other.x) &&
                        (y == other.y) &&
                        (z == other.z);
            }

            void operator+= (float3 other) {
                x += other.x;
                y += other.y;
                z += other.z;
            }

            void operator-= (float3 other) {
                x -= other.x;
                y -= other.y;
                z -= other.z;
            }

            std::string to_string() {
                return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
            }
        };
    }
}

// Allow inclusion into std::set
namespace std {
    template <> struct hash<ShapeDescriptor::cpu::float3>
    {
        size_t operator()(const ShapeDescriptor::cpu::float3& p) const
        {
            return std::hash<float>()(p.x) ^ std::hash<float>()(p.y) ^ std::hash<float>()(p.z);
        }
    };
}


inline ShapeDescriptor::cpu::float3 make_float3_cpu(float x, float y, float z) {
    ShapeDescriptor::cpu::float3 out;
    out.x = x;
    out.y = y;
    out.z = z;
    return out;
}

inline float length(ShapeDescriptor::cpu::float3 vec) {
    return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

inline ShapeDescriptor::cpu::float3 normalize(ShapeDescriptor::cpu::float3 in) {
    ShapeDescriptor::cpu::float3 out;
    float len = length(in);
    out.x = in.x / len;
    out.y = in.y / len;
    out.z = in.z / len;
    return out;
}

inline ShapeDescriptor::cpu::float3 operator* (ShapeDescriptor::cpu::float3 vec, float other) {
    ShapeDescriptor::cpu::float3 out;
    out.x = vec.x * other;
    out.y = vec.y * other;
    out.z = vec.z * other;
    return out;
}

inline ShapeDescriptor::cpu::float3 operator*(float other, ShapeDescriptor::cpu::float3 vec) {
    return operator*(vec, other);
}

inline std::ostream & operator<<(std::ostream & os, ShapeDescriptor::cpu::float3 vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}
